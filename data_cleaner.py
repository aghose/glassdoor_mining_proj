# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:27:22 2020

@author: Akash Ghose
"""

import pandas as pd
import re

data = 'uncleaned_data.csv'
    
def clean_data(data, keywords= []):
    uncleaned_df = pd.read_csv(data)
    
    #Removes columns where all values are -1s 
    intermediate_df = uncleaned_df.loc[:, (uncleaned_df != -1).any(axis = 0)]
    
    #If the user passes in a list of keywords,
    #Get rid of any rows that does not contain the keywords in their 'Job Title'
    if keywords:
        df = intermediate_df['Job Title']
        filtered_job_titles = filter(my_filter, [keywords, df])    
    
    
    final_df = {}
    return final_df


def my_filter(obj):
    print(obj)
    
    kwds = obj[0]
    dfs = obj[1]
    
    print("Keywords: "+kwds)
    dfs.head()

clean_data('uncleaned_data.csv', keywords='Data Scientist')
