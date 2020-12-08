# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:27:22 2020

@author: Akash Ghose
"""

import pandas as pd
#import numpy as np

#data = pd.read_csv('uncleaned_data.csv')

"Goals for data cleaning:"
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
    
def clean_data(data_file_path, current_year):
    
    data = pd.read_csv(data_file_path)
    
    df = data.drop('Headquarters',1)
    df = df.drop('Competitors', 1)
    
    #Drops rows with which have Job Description as -1
    df = df[df['Job Description'] != '-1']
       
    "Salary Parsing"
    
    df_missing_salary = df[df['Salary Estimate'] == '-1']
    df = df[df['Salary Estimate'] != '-1']
    #Drops "(Glassdoor estimate)"
    salary = df["Salary Estimate"].apply(lambda x: x.lower().split('(')[0])
    #Drops "K" and "$"
    salary = salary.apply(lambda x: x.replace("k","").replace("$",""))
    #Drop "per hour" and "employer provided"
    salary = salary.apply(lambda x: 
                          x.replace('per hour', '').
                          replace('employer provided salary:',''))    
    
    #appends 'per hour' and 'employer provided salary:' as features
    df['hourly'] = df['Salary Estimate'].apply(
        lambda x: 1 if 'per hour' in x.lower() else 0)
    df['employer_provided_salary'] = df['Salary Estimate'].apply(
        lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
    #Gets min
    df["min_salary"] = salary.apply(lambda x: int((x.split('-')[0])))
    #Gets max
    df['max_salary'] = salary.apply(lambda x: int(x.split('-')[1]))
    #Gets avg
    df['avg_salary'] = (df.min_salary + df.max_salary)/2
    
    "Company Name Parsing"
    #Drops the ratings attached to the end of the company names
    company_name = df.apply(
        lambda x: x['Company Name'] if x.Rating <0 else 
        x['Company Name'][:-3], axis=1)
    df['Company Name'] = company_name
    
    #State parsing
    df['location_state'] = df.apply(lambda x: x.Location.split(',')[1],axis=1)
    
    #Company age
    #current_year = 2020
    df['age'] = df.Founded.apply(lambda x: x if x <0 else current_year-x)
    
    #Check Job description to see if it contains keywords
    df['contains_python'] = df['Job Description'].apply(
        lambda x: 1 if 'python' in x.lower() else 0)
    df['contains_R'] =  df['Job Description'].apply(
        lambda x: 1 if ' r ' in x.lower() or 
        'r-studio' in x.lower() or 
        'r studio' in x.lower() else 0)
    df['contains_big_data'] = df['Job Description'].apply(
        lambda x: 1 if 'big data' in x.lower() or 
        'spark' in x.lower() or 
        'hadoop' in x.lower() else 0)
    df['contains_MS'] = df['Job Description'].apply(
        lambda x: 1 if ' ms ' in x.lower() or 
        "master's" in x.lower() or
        "masters'" in x.lower() or 
        "masters" in x.lower() or 
        'master' in x.lower() else 0)
    
    #Re-ordering columns for ease of readability
    df = df[['Job Title', 'Salary Estimate', 'min_salary', 'max_salary',
       'avg_salary', 'hourly', 'employer_provided_salary', 'Job Description',
       'contains_python', 'contains_R', 'contains_big_data', 'contains_MS',
       'Rating', 'Company Name', 'Location', 'location_state', 'Size',
       'Founded', 'age', 'Type of ownership', 'Industry', 'Sector', 'Revenue',
       ]]
    
    final_df = df
    df_missing_salary.to_csv('missing_salary.csv', index = False)
    return final_df
