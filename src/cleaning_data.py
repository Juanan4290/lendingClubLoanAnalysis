# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 08:19:17 2018

@author: Juan Antonio Morales
"""

### 1. libraries ###
import warnings # remove warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


### 2. read data ###
print("Reading Data...")

loans = pd.read_csv("/media/juanan/DATA/loan_data_analysis/data/raw/accepted_2007_to_2017Q3.csv.gz")


### 3. cleaning data ###
print("Starting Cleaning Data...")

## filter only Fully Paid and Charged Off loans
loans = loans[(loans['loan_status'] == "Fully Paid") | (loans['loan_status'] == "Charged Off")]
# target to integer
loan_status_dict = {'Fully Paid': 0,
                    'Charged Off': 1}

loans['loan_status'] = loans['loan_status'].map(lambda i: loan_status_dict[i])

## removing columns with 25% of NA's
loans[loans == ""] = np.nan
na_percentage = loans.isnull().sum()/loans.shape[0] #number of NA's per column

loans = loans.loc[:,na_percentage < 0.25] 

## removing features with just one value
number_of_unique_values_per_column = loans.apply(lambda i: i.nunique(), axis = 0)
loans = loans.loc[:,number_of_unique_values_per_column != 1]

## selecting variables
print("Selecting variables...")

columns_of_interest = ['num_bc_sats', 'num_rev_tl_bal_gt_0', 'grade', 'avg_cur_bal', 'pub_rec_bankruptcies', 
                       'num_rev_accts', 'tax_liens', 'funded_amnt_inv', 'delinq_2yrs', 'total_bal_ex_mort',
                       'pct_tl_nvr_dlq', 'disbursement_method', 'fico_range_low', 'verification_status', 'delinq_amnt',
                       'purpose', 'loan_amnt', 'installment',
                       'fico_range_high', 'annual_inc', 'term', 'int_rate', 'emp_length',
                       'revol_bal', 'application_type', 'num_bc_tl', 'num_sats', 'tot_hi_cred_lim', 
                       'tot_coll_amt', 'initial_list_status', 'bc_open_to_buy', 'total_bc_limit', 
                       'open_acc', 'revol_util', 'pub_rec', 'funded_amnt', 'num_il_tl', 
                       'addr_state', 'num_accts_ever_120_pd', 'total_il_high_credit_limit', 'bc_util', 'percent_bc_gt_75', 
                       'sub_grade', 'mort_acc', 'num_op_rev_tl', 'dti', 'home_ownership', 'loan_status']

loans = loans[sorted(columns_of_interest)]


### 4. processing numeric variables ###
print("Processing numeric variables...")

numeric_variables = loans._get_numeric_data().columns
## fill NA's to median
loans[numeric_variables] = loans[numeric_variables].apply(lambda i: i.fillna(i.median()), axis = 1)


### 5. processing categorical variables ###
print("Processing categorical variables...")

categorical_variables = loans.select_dtypes(include="object").columns
## fill NA's to last valid observation
loans[categorical_variables] = loans[categorical_variables].fillna(method = "ffill")


### 6. writing clean data ###
print("Writing data to disk...")
# local directory: "/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv"
# relative directory: "../data/loans_processed.csv"
print("Final dataset with {} rows and {} columns".format(loans.shape[0], loans.shape[1]))
loans.to_csv("/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv", sep = "^", index = False)
print("Done!")
