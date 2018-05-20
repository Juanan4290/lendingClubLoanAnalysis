#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: juanan
"""

### 01 - Libraries ############################################################

import pandas as pd
import os

import matplotlib.pyplot as plt; plt.style.use("ggplot")

from process_loan_status import *

### 02 - Read the data ########################################################

os.chdir('/home/juanan/Documentos/ja-github/loan-data-analysis/src/processing')

loans = pd.read_csv("../data/clean/loans.csv", sep = "^")


### 03 - Process Target #######################################################

loans['loan_status'] = loans['loan_status'].map(process_loan_status)

loans = loans[loans['loan_status']<2]


### 04 - Writing our sample data 

#loans.to_csv('../../data/loans_pro.csv', sep = "^", index=False)

