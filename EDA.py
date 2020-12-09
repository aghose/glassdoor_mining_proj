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

import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv

data = pd.read_csv('cleaned_data.csv')

description = data.describe()

profile = ProfileReport(data,title='Your EDA Report')
profile.to_file("EDA_profile_report.html")

report_sweetviz = sv.analyze(data)
report_sweetviz.show_html()
