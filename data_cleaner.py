# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:27:22 2020

@author: Akash Ghose
"""

import pandas as pd
#import numpy as np

#data = pd.read_csv('uncleaned_data.csv')

"clean_data will:"
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
        #Convert hourly wages into salary
    #Company name parsing:
        #Company name only - get rid of everything else in that field 
    #State field ##
    #Age of company ##
    #Get job description length and put it in as a feature
    #Parsing of job description (python, R, etc.) ##
    #Parsing job titles:
        #Simplifying job titles:
            # Using a list of keywords, simplify the titles so they can be
            # grouped together. i.e titles like 
            # "Research Data Scientist", "Chemistry Data Scientist",
            # "Buisness Data Scientist" etc. will all have the simplified title
            # of "Data Scientist"
        #Parsing for seniority:
            #Parse the titles for keywords to indicate seniority level of the position
    
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
    #convert hourly to yearly salary
    df['min_salary'] = df.apply(lambda x: x.min_salary if x.hourly==0 else x.min_salary*2, axis=1)
    df['max_salary'] = df.apply(lambda x: x.max_salary if x.hourly==0 else x.max_salary*2, axis=1)
    df['avg_salary'] = df.apply(lambda x: x.avg_salary if x.hourly==0 else x.avg_salary*2, axis=1)
    
    "Company Name Parsing"
    
    #Drops the ratings attached to the end of the company names
    company_name = df.apply(
        lambda x: x['Company Name'] if x.Rating <0 else 
        x['Company Name'][:-3], axis=1)
    df['Company Name'] = company_name
    
    "State parsing"
    
    df['location_state'] = df.apply(lambda x: x.Location.split(',')[1],axis=1)
    df['location_state'] = df['location_state'].apply(
        lambda x: ' CA' if 'los angeles' in x.lower() else x)    
    
    'Company age'
    #current_year = 2020
    df['age'] = df.Founded.apply(lambda x: x if x <0 else current_year-x)
    
    'Parse Job descrpition'
    
    df['description_length'] = df['Job Description'].apply(lambda x: len(x))
    
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
    
    "Parsing Job Titles"
    
    #Will add a simplified job title column
    df['simplified_title'] = df['Job Title'].apply(simplify_job_title)
    
    #This will parse job title for info about senority and add it as a column
    df['seniority'] = df['Job Title'].apply(find_seniority)
    
    #Re-ordering columns for ease of readability
    df = df[['Job Title','simplified_title', 'seniority', 'Salary Estimate', 
             'min_salary', 'max_salary', 'avg_salary', 'hourly', 
             'employer_provided_salary', 'Job Description', 'description_length',
             'contains_python', 'contains_R', 'contains_big_data', 'contains_MS',
             'Rating','Company Name', 'Location', 'location_state', 'Size', 'Founded',
             'age', 'Type of ownership', 'Industry', 'Sector', 'Revenue']]
    
    final_df = df
    df_missing_salary.to_csv('missing_salary.csv', index = False)
    return final_df

def convert_hourly(df):
    
    return df

def simplify_job_title(title):
    if all(x in title.lower() for x in ['data', 'scientist']):
        return 'Data Scientist'
    elif (all(x in title.lower() for x in ['research', 'scientist']) or
          all(x in title.lower() for x in ['r&d', 'scientist'])):
        return 'Research Scientist'
    elif (all(x in title.lower() for x in ['data', 'analyst']) or
          all(x in title.lower() for x in ['data', 'analytics'])):
        return 'Data Analyst'
    elif (all(x in title.lower() for x in ['research', 'analyst']) or
          all(x in title.lower() for x in ['r&d', 'analyst'])):
        return 'Research Analyst'
    elif all(x in title.lower() for x in ['data', 'engineer']):
        return 'Data Engineer'
    elif all(x in title.lower() for x in ['machine', 'learning']):
        return 'Machine Learning Engineer'
    elif 'scientist' in title.lower():
        return 'Other Scientist'
    elif 'analyst' in title.lower():
        return 'Other Analyst'
    elif ('instructor' in title.lower()):
        return 'Instructor'
    else: return 'Other'
    
def find_seniority(title):
    if any(x in title.lower() for x in ['sr', 'senior', 'manager', '3', 
                                        'lead', 'principal', ' iii']):
        return 'Senior'
    elif any(x in title.lower() for x in['mid', 'ii', '2']):
        return 'Mid-level'
    elif any(x in title.lower() for x in ['jr', 'junior', ' i ', '1', 'associate', 'entry']):
        return 'Junior'
    else:
        return 'na'

