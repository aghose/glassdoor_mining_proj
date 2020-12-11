# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 08:33:48 2020

@author: asgho
"""

import pandas as pd

data = pd.read_csv('cleaned_data.csv')

df = data[['avg_salary', 'description_length', 'Rating', 'age',
           'simplified_title','seniority', 'hourly', 'employer_provided_salary',
           'contains_python','contains_R', 'contains_big_data', 'contains_MS',
           'Location', 'location_state', 'Size', 'Type of ownership',
           'Industry', 'Sector', 'Revenue', 'revenue_num','size_num']]

#Create boolean columns of all the categorical data other than revenue and size
columns=["simplified_title", 'seniority',
         'location_state','Type of ownership','Sector']
dum_df = pd.get_dummies(df[columns], columns=["simplified_title", 'seniority',
                                     'location_state','Type of ownership',
                                     'Sector'])

# Merge the two data farmes
df_merge = df.join(dum_df)
