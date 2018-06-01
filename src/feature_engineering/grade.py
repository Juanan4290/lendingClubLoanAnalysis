# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:20:14 2018

@author: Juan Antonio Morales
"""

def feature_grade(data):
    
    data['high_risk'] = ((data['grade'] == 'F')|(data['grade'] == 'G')|(data['grade'] == 'E'))
    data['medium_risk'] = ((data['grade'] == 'C')|(data['grade'] == 'D'))
    data['low_risk'] = ((data['grade'] == 'A')|(data['grade'] == 'B'))
    
    grade_dict = {'A': 1,
                  'B': 2,
                  'C': 3,
                  'D': 4,
                  'E': 5,
                  'F': 6,
                  'G': 7}
    
    data['numeric_grade'] = data['grade'].map(lambda i: grade_dict[i])
    
    return data