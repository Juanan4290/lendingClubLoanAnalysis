#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: juanan
"""

### 01 - Libraries ############################################################

import pandas as pd

import matplotlib.pyplot as plt; plt.style.use("ggplot")

from src.processing.process_loan_status import process_loan_status

### 02 - Read the data ########################################################

loans = pd.read_csv("./data/clean/loans.csv", sep = "^")


### 03 - Process Target #######################################################

loans['loan_status'] = loans['loan_status'].map(process_loan_status)

loans = loans[loans['loan_status']<2]


### 04 - Writing our sample data 

#loans.to_csv('../../data/loans_pro.csv', sep = "^", index=False)

