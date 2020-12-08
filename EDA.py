# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:33:52 2020

@author: Akash Ghose
"""

#Here I will spend a little time reading and exploring the data I've gathered
#I had to install the following packages:
    #pandas-profiling
    #sweetviz

import pandas as pd
#from pandas_profiling import ProfileReport
#import sweetviz as sv

data = pd.read_csv('uncleaned_data.csv')

#profile = ProfileReport(data,title='Your EDA Report')
#profile.to_file("EDA_profile_report.html")

#report_sweetviz = sv.analyze(data)
#report_sweetviz.show_html()
