# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:59:21 2020

@author: Akash Ghose
"""

import data_cleaner as dc

'''
    Requirements: csv file path
    
    Output:
        Will remove all columns that contain only the values of -1s
'''
dc.clean_data('uncleaned_data.csv')