#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:13:58 2018

@author: juanan
"""

def process_loan_status(loan_status):
    
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