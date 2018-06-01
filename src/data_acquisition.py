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

# Remove duplicates:
loans = loans.drop_duplicates()

### 3. write data ###

loans.to_csv("/media/juanan/DATA/loan_data_analysis/data/raw/accepted_2007_to_2017Q3.csv.gz", compression="gzip",
             sep = "^", index=False)