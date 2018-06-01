#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:04:10 2018

@author: Juan Antonio Morales
"""

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler, RobustScaler

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