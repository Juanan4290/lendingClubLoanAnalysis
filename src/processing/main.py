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
from src.processing.process_numerical_variables import process_numerical_variables


if __name__ == '__main__':

    ### 02 - Read the data ####################################################
    
    loans = pd.read_csv("../../data/clean/loans.csv", sep = "^").sample(200000, random_state=4290)
    
    
    ### 03 - Processing variables #############################################
    
    loans = process_loan_status(loans)
    loans = process_numerical_variables(loans)
    
    ### 04 - Writing our sample data 
    
    #loans.to_csv('../../data/loans_pro.csv', sep = "^", index=False)

