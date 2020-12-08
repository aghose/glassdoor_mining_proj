# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:59:21 2020

@author: Akash Ghose
"""

import data_cleaner as dc

'''
    Requirements: csv file path
    
    Output:
        #Drop the following features: ##
            #Headquarters ##
            #Competitors ##
        #Drop rows where 'Job Description' == -1
        #Salary parsing: ##
            #Drop rows with missing salary vals and store them elsewhere ##
            #Drop "(Glassdoor estimate)" ##
            #Drop "K" and "$" ##
            #Drop "per hour" and "employer provided" -> replace and append as features ##
            #Get min, max and avg
        #Company name parsing:
            #Company name only - get rid of everything else in that field 
        #State field ##
        #Age of company ##
        #Parsing of job description (python, R, etc.) ##
'''
df = dc.clean_data('uncleaned_data.csv', 2020)

df.to_csv('cleaned_data.csv', index=False)
