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


def parse_emp_title(emp_title, number_of_occurrences, n = 10):
    """
    Parameters
    ----------
    emp_title: employee title string
    number_of_occurrences: number of occurrences grouped by employee title
    n: number of occurences cut-off where employee title will parse to "other"
    
    Returns:
    ---------
    emp_title: old employee title sring
    result: new employee title string
    """
    
    if number_of_occurrences > n:
        result = emp_title.lower().strip()
    else:
        result = "other"
    
    return [emp_title,result]


def process_emp_title(data, n = 10):
    """
    Parameters
    ----------
    data: DataFrame to process emp_title variable
    n: number of occurences cut-off where employee title will parse to "other"
    
    Returns:
    ----------
    result: emp_title variable processed
    """
    
    # number of occurrences by employee title
    emp_title_counter = data.groupby("emp_title")["loan_status"].agg("count").reset_index()
    emp_title_counter.columns = ["emp_title", "number_of_occurrences"]
    
    # parsing "parse_emp_title" function and transform to dictionary
    emp_title_dict = emp_title_counter.apply(lambda i: parse_emp_title(i[0], i[1], n), 
                                             axis = 1).set_index("emp_title")
    emp_title_dict = emp_title_dict["number_of_occurrences"].to_dict()
    
    result = data["emp_title"].map(lambda i: emp_title_dict[i])
    
    return result


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