### 01 - Libraries #######################################################

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd


### 02 - Load Data #######################################################

os.chdir("/home/juanan/Documentos/ja-github/loan-data-analysis/src/")

raw_data_path = "../data/raw"

list_dir = os.listdir(raw_data_path)

list_ = []
for file in list_dir:
    full_path = os.path.join(raw_data_path, file)
    df = pd.read_csv(full_path, sep = ",")
    list_.append(df)
    
loans = pd.concat(list_)


### 03 - Clean Data ######################################################

columns_of_interest = ["funded_amnt_inv", "term", "issue_d", "installment", "int_rate", 
                       "grade", "emp_title", "emp_length", "annual_inc", "title",
                       "dti", "home_ownership", "zip_code", "addr_state","total_rec_late_fee", 
                       "application_type", "total_acc", "loan_status"]


loans = loans[columns_of_interest]

na_count = loans.isnull().sum() / loans.shape[0]

loans = loans.dropna(axis = 0, how = "all")

# Remove duplicates:
loans = loans.drop_duplicates()

### Write Data ###########################################################

loans.to_csv("../data/clean/loans.csv", sep = "^", index=False)