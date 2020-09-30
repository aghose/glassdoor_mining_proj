# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:27:22 2020

@author: Akash Ghose
"""

import pandas as pd


data = 'uncleaned_data.csv'
    
def clean_data(data, keywords= []):
    
    #First order of buisness, read the data that needs to be cleaned
    uncleaned_df = pd.read_csv(data)
    
    #Removes columns where all values are -1s 
    intermediate_df = uncleaned_df.loc[:, (uncleaned_df != -1).any(axis = 0)]
    
    #Get rid of any rows that does not contain the keywords in their 'Job Title'
    if keywords:
        print("keywords is not empty")
    
    
    
    
    final_df = {}
    return final_df
