# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:33:52 2020

@author: Akash Ghose
"""

#Here I will spend a little time reading and exploring the data I've gathered
#I had to install the following packages:
    #pandas-profiling
    #sweetviz

"Goals"
    #Draw histograms
    #Draw boxplots
    #Draw barplots
    #Make WordCloud
    #Draw pivot_tables
    #What I'm really interested in is how the otehr variables interact with salary.
        #Which states have the highest salaries? Which locations?
        #How does rating correspond with it? descrpt_lenght?
    
    #Plot revenue vs salary vs job title?
        #Pivot table?
        #Plot revnue on one axis and salary on the other? Fit trend?
        #Color code plot points by jobtitle?
    #Pivot tables with salary and job title
    
    #convert catagorical data i.e size and revenue to numerical. --sklearns
    #Explore correlations graphically.
        #Remove outliers/normalize data
        
    #lm(seniority, job title, rating, sector, type of ownership)
    #Lasso regression
    #Kmeans to find outliers?
        #Cluster Dendrogams? Hierarchical clustering?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import sweetviz as sv

data = pd.read_csv('cleaned_data.csv')

#Setting console display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

description = data.describe()

# profile = ProfileReport(data,title='Your EDA Report')
# profile.to_file("EDA_profile_report.html")

# report_sweetviz = sv.analyze(data)
# report_sweetviz.show_html()

# avg salary across all entries
data.avg_salary.mean()

#Show what percentages contain r vs python vs big data etc.
data.contains_python.value_counts(normalize=True).plot(kind='bar')
data.contains_R.value_counts(normalize=True).plot(kind='bar')
data.contains_big_data.value_counts(normalize=True).plot(kind='bar')
data.contains_MS.value_counts(normalize=True).plot(kind='bar')

df = data[['avg_salary', 'description_length', 'Rating', 'age',
           'simplified_title','seniority', 'hourly', 'employer_provided_salary',
           'contains_python','contains_R', 'contains_big_data', 'contains_MS',
           'Location', 'location_state', 'Size', 'Type of ownership',
           'Industry', 'Sector', 'Revenue', 'revenue_num', 'revenue_num_unknown',
           'size_num', 'size_num_unknown','simplified_title_num', 
           'seniority_num', 'location_state_num','type_of_ownership_num', 'sector_num']]
# Seperating the data out into continuous and discrete variables
df_continuous = data[['avg_salary', 'description_length', 'Rating', 'age']]
df_discrete = data[['simplified_title','seniority', 'hourly', 
                        'employer_provided_salary', 'contains_python',
                        'contains_R', 'contains_big_data', 'contains_MS',
                        'Location', 'location_state', 'Size', 
                        'Type of ownership', 'Industry', 'Sector', 'Revenue',
                        'revenue_num', 'revenue_num_unknown', 'size_num', 
                        'size_num_unknown','simplified_title_num', 
                        'seniority_num', 'location_state_num',
                        'type_of_ownership_num', 'sector_num']]

# title vs salary
table_00 = (pd.pivot_table(data, index=['simplified_title'], 
                        values= ['avg_salary']).
            sort_values(by='avg_salary', ascending=False))

table_00

# [title and seniority] vs salary
table_01 = (pd.pivot_table(data, index=['simplified_title', 'seniority'], 
                        values= ['avg_salary']))
table_01

# [title and revenue] vs salary
table_02 = (pd.pivot_table(data, values='avg_salary', 
                           index=['simplified_title', 'Revenue']))

# Plot revnue on one axis and salary on the other
sns.set_theme(style="whitegrid")
# ax = sns.violinplot(x="Revenue", y="avg_salary", data=data,
# #                   inner=None, color=".8")
ax = sns.stripplot(x="revenue_num", y="avg_salary", data=data)
ax = sns.boxplot(x="revenue_num", y="avg_salary", data=data, whis=np.inf)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# Correlations
df_corr = df.corr()
data_corr = data.corr()
# plt.figure(figsize=(5,5))
sns.heatmap(df_corr,
            vmin=-1,
            cmap='coolwarm')

plt.figure(figsize=(30,20))
sns.heatmap(data_corr,
            vmin=-1,
            cmap='coolwarm')

df.plot.scatter(x='Rating', y='avg_salary')

df.plot.scatter(x='age', y='description_length')
