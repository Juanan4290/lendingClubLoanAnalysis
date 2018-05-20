#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:04:10 2018

@author: juanan
"""
import numpy as np
from collections import Counter

def loan_status_to_numeric(loan_status):
    """
    Takes a loan status and returns a numeric code according to the
    loan status dictionary: 
        0 for fully paid loans
        1 for unpaid or charge off loans
        2 for current loans
    """
    
    loan_status_dict = {
    "Current": 2,
    "Fully Paid": 0,
    "Charged Off": 1,
    "Late (31-120 days)": 2,
    "In Grace Period": 2,
    "Late (16-30 days)": 2,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Default": 1
    }
    
    return loan_status_dict[loan_status]

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = [k for k, v in outlier_indices.items() if v > n]
    
    return multiple_outliers