#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:18:42 2018

@author: juanan
"""
from src.utils import detect_outliers

def process_numerical_variables(loans):
    """
    Takes loan dataframe, process numerical varaibles and returns loans dataframe
    """
    
    # list of numerical variables
    numerical_variables = ["funded_amnt_inv", "installment", "int_rate", "annual_inc", "dti",
                           "total_rec_late_fee", "total_acc"]
    
    # transform interest rate to numeric
    loans['int_rate'] = loans['int_rate'].map(lambda x: float(x[:-1]))
    
    # fill na with the median
    loans[numerical_variables] = loans[numerical_variables]\
                                     .fillna(loans[numerical_variables]\
                                             .median())
    
    return loans