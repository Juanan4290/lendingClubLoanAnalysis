#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:13:58 2018

@author: juanan
"""
from src.utils import loan_status_to_numeric

def process_loan_status(loans):
    """
    Takes loan dataframe, process loan_status and returns loans dataframe
    """
    
    # transform loan_status string to numeric
    loans['loan_status'] = loans['loan_status'].map(loan_status_to_numeric)

    # keep only unpaid loans (1) and fully paid loans (0)    
    loans = loans[loans['loan_status'] < 2]
    
    return loans