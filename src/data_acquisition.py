### 1. libraries ###

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd


### 2. Data Acquisition ###
print("Starting Data Acquisition...")
os.chdir("/media/juanan/DATA/loan_data_analysis/data/raw")

raw_data_path = "../data/raw"

list_dir = os.listdir(raw_data_path)

list_ = []
for file in list_dir:
    full_path = os.path.join(raw_data_path, file)
    df = pd.read_csv(full_path, sep = ",")
    list_.append(df)
    
loans = pd.concat(list_)

## selecting variables
print("Selecting variables...")

columns_of_interest = ['num_bc_sats', 'num_rev_tl_bal_gt_0', 'grade', 'avg_cur_bal', 'pub_rec_bankruptcies', 
                       'num_rev_accts', 'tax_liens', 'funded_amnt_inv', 'delinq_2yrs', 'total_bal_ex_mort',
                       'pct_tl_nvr_dlq', 'disbursement_method', 'fico_range_low', 'verification_status', 'delinq_amnt',
                       'purpose', 'emp_title', 'zip_code', 'loan_amnt', 'installment',
                       'fico_range_high', 'annual_inc', 'term', 'int_rate', 'emp_length',
                       'revol_bal', 'application_type', 'num_bc_tl', 'num_sats', 'tot_hi_cred_lim', 
                       'tot_coll_amt', 'initial_list_status', 'bc_open_to_buy', 'total_bc_limit', 
                       'open_acc', 'revol_util', 'pub_rec', 'funded_amnt', 'num_il_tl', 
                       'addr_state', 'num_accts_ever_120_pd', 'total_il_high_credit_limit', 'bc_util', 'percent_bc_gt_75', 
                       'sub_grade', 'mort_acc', 'num_op_rev_tl', 'dti', 'home_ownership', 'loan_status']

loans = loans[sorted(columns_of_interest)]

# Remove duplicates:
loans = loans.drop_duplicates()

### 3. write data ###
loans.to_csv("/media/juanan/DATA/loan_data_analysis/data/clean/loans.csv.gz", compression="gzip")