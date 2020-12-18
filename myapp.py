# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:37:30 2020

@author: Akash Ghose
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm # import statsmodels
from sklearn import linear_model
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import locale
locale.setlocale(locale.LC_ALL, '')  # Use '' for auto, or force e.g. to 'en_US.UTF-8'
from numpy import arange
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
import altair as alt

import numpy as np
sns.set_style("darkgrid")

st.sidebar.title("Operations on the App")

#st.subheader("Checkbox")
app_only = st.sidebar.checkbox("Show the app only", False)
show_summary = st.sidebar.checkbox("Show Summary", True)
show_report = st.sidebar.checkbox("Show Report", True)

if app_only:
    show_summary = False
    show_report = False
    show_data = False

# I am loading the functions related to reading and processing my data first
# So that I can have my data and processed template as global variables        
@st.cache
def read_data():
    data = pd.read_csv('cleaned_data.csv')
    return data

@st.cache
def preprocessing(data):
    df = data[['avg_salary', 'description_length', 'Rating', 'age',
           'simplified_title','seniority', 'hourly', 'employer_provided_salary',
           'contains_python','contains_R', 'contains_big_data', 'contains_MS',
           'location_state', 'Size', 'Type of ownership',
           'Sector', 'Revenue', 'revenue_num','size_num']]
    #Create boolean columns of all the categorical data other than revenue and size
    columns=["simplified_title", 'seniority',
             'location_state','Type of ownership','Sector']
    # I am selecting just df[columns] instead of df so as to avoid overlap when 
    # I try to merge the two dataframes back together again
    dum_df = pd.get_dummies(df[columns], columns=["simplified_title", 'seniority',
                                         'location_state','Type of ownership',
                                         'Sector'])
    
    # Merge the two data farmes
    df_merge = df.join(dum_df)
    
    # Drop all columns that are not numbers
    for col in df_merge.columns: 
        if df_merge[col].dtype=='object': 
            print("deleted: " +str(col))
            del df_merge[col]
    
    #predictors
    df_predictors = df_merge.loc[:, df_merge.columns != 'avg_salary']
    #Target -> avg_salary
    df_target = df['avg_salary'].values
    
    x = df_predictors
    y = df_target
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)
    return x_train, x_test, y_train, y_test

@st.cache
def fix_template(template, data):
    #Set description_length, rating, age, revenue and size to the median description_length
    template.description_length = data.description_length.median()
    template.Rating = data.Rating.median()
    template.age = data.age.median()
    template.revenue_num = data.revenue_num.median()
    template.size_num = data.size_num.median()  
    
    #Set everything else to 0
    template = pd.DataFrame(myFunc(template))
    return template

def myFunc(template):
    columns = ['description_length','Rating','age','revenue_num','size_num']
    for column in template.columns.difference(columns):
        template[column] = 0
    return template

@st.cache
def make_template(x_train, data):
    template = pd.DataFrame([x_train.iloc[0]])
    template = fix_template(template, data)
    return template

 
def eda():
    st.subheader("The average salary for data science jobs accross the US is:")
    st.write('$',(data.avg_salary.mean() * 1000))
    
    st.markdown("""**Fraction of job descriptions that contained Python:**""")
    st.bar_chart(data.contains_python.value_counts(normalize=True))
    st.markdown("""**Fraction of job descriptions that contained R:**""")
    st.bar_chart(data.contains_R.value_counts(normalize=True))
    st.markdown("""**Fraction of job descriptions that mentioned Big Data:**""")
    st.bar_chart(data.contains_big_data.value_counts(normalize=True))
    st.markdown("""**Fraction of job descriptions that mentioned a Masters Degree:**""")
    st.bar_chart(data.contains_MS.value_counts(normalize=True))
    
    st.subheader('Title vs Salary')
    table_00 = (pd.pivot_table(data, index=['simplified_title'], 
                            values= ['avg_salary']).
                sort_values(by='avg_salary', ascending=False))
    plot00 = table_00.plot.bar()
    fig00 = plot00.get_figure()
    st.pyplot(fig00)
    
    st.subheader('[Title and Seniority] vs salary')
    table_01 = (pd.pivot_table(data, index=['simplified_title', 'seniority'], 
                            values= ['avg_salary']).
                sort_values(by=['seniority','avg_salary'], ascending=False))
    
    plot01 = table_01.plot.bar()
    fig01 = plot01.get_figure()
    st.pyplot(fig01)
    fig01.clear()
    
        
    st.subheader('State vs Salary')
    state_salary = (pd.pivot_table(data, index='location_state', values='avg_salary').
                    sort_values(by='avg_salary', ascending=True))
    plot_st_sl = state_salary.plot.barh()
    fig_st_sl = plot_st_sl.get_figure()
    st.pyplot(fig_st_sl,width=100,height=1000)
    fig_st_sl.clear()
    
    st.subheader('States vs Job listings')
    listings_state = data.location_state.value_counts().sort_values(ascending=True)
    plot_st_listings = listings_state.plot.barh()
    fig_st_listings = plot_st_listings.get_figure()
    st.pyplot(fig_st_listings)

    st.subheader('Data Correlations')
    data_corr = data.corr()
    fig03 = plt.figure()
    sns.heatmap(data_corr,
                vmin=-1,
                cmap='coolwarm')
    
    st.pyplot(fig03)
    
    
    
#Load the data
data = read_data()

#Process the data and template
x_train, x_test, y_train, y_test = preprocessing(data)
    
template = make_template(x_train, data)

if show_summary:
    st.markdown("""
    ## **Summary**:
    
    I have collected data from 2000 [Glassdoor](https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword=Data_Scientist&locT=&locId=0&jobType=&context=Jobs&sc.keyword=Data_Scientist&dropdown=0) job posting regarding Data Scientists. I have cleaned, processed and analyzed that data, and afterwards, I applied machine learning algorithms on that data to build a data product that can estimate a salary, given certain variables. The app is at the bottom of the page
    
    
                """)

if show_report:
    st.title('Glassdoor Mining Project')
    
    st.markdown("""
                
    ## Technologies used:
    
    Python, pandas, numpy, seaborn, selenium, matplotlib, streamlit, Rmarkdown, KnitR and Heroku
    
    ## How to use my project:
    
    You may access my final product here: <https://glassdoor-mining-proj.herokuapp.com/>
    
    If you wish to run my files, the entire project can be found here: <https://github.com/aghose/glassdoor_mining_proj>
    
    From my github page, you may download all the files. If you wish to run my scraper, run the run_scraper.py file
    If you wish to run my data cleaner, run my run_cleaner.py file
    If you wish to see the entirety of my EDA, run EDA.py
    My models are made and stored in modeling.py
    If you wish to run my data app locally, from your terminal, run the command "streamlit run myapp.py"
      * You need to in the project directory in the terminal
      * You need to have streamlit installed
    
    ---
    
    ## Phase 01: Project Planning
    
    **Area of research:** Glassdoor Job Market.
    
    **Title of project:** Glassdoor Data analysis and Salary estimator.
    
    **Potential clients:** Students or people looking for jobs.
    
    **Potential sponsors:** People who wished scrape Glassdoor and anaylize the job market for a given job.
    
    **Objective:** Sometime back in October, I was on [Glassdoor](https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword=Data_Scientist&locT=&locId=0&jobType=&context=Jobs&sc.keyword=Data_Scientist&dropdown=0) looking at job postings for Data Scientists. As I browsed through them, I had several questions. I wondered how many of those jobs required Python vs R. Or how many wanted you to have a masters degree. I wondered which States paid you the most. I wondered if there was difference in average pay amongst the different job titles I noticed. I also noticed that some of the job listings did not have any Glassdoor estimates for salaries. I wondered if it might be possible for me to be able to *predict* some of these estimates. It seemed like a reach at that time, given my level of knowledge and experience, but I thought, why not try. 

    ## Phase 02: Data Collection
    
    First, I tried searching for a public API to access Glassdoor's data. Unfortunately, they didn't have any public APIs. The quest for me did not end here however; I had discovered that I was not the first one to have the idea to mine Glassdoor for a project, and in fact, someone had already built a [Glassdoor scraper!](https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905). While I was down this rabbit hole of research, trying to gather materials and info on how to complete this seemingly daunting (at that time) task I had set out to do, I ran across Ken Jee's YouTube channel. There he talks about doing things very similar to what I had in mind. I used [his video](https://www.youtube.com/watch?v=GmW4F6MHqqs&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t&index=2&ab_channel=KenJee) and the scrapper I mentioned to help me get started and collect my own data! I collected 2,000 entries, stored them in a [csv](./uncleaned_data.csv)
    
    ## Phase 03 & 04: Data Cleaning & EDA
    
    I decided to put these two phases together here, because while I was doing the project, there was not really a clear distinction between these two phases; One was driven by the other. I would my data in anticipation of EDA I wanted to perform, go over to my EDA file, look at my data, and then come back to my cleaner file and further clean my file. I repeated this process several times until I was somewhat satisfied. 
    
    Here are some steps I took to clean my data:
    
      * Drop the following features because the contained no data:
      
          + Headquarters 
    
          + Competitors
      
      * Drop rows where 'Job Description' == -1
      
      * Salary parsing: 
      
          + Drop rows with missing salary vals and store them elsewhere 
      
          + Drop "(Glassdoor estimate)" 
      
          + Drop "K" and "$" 
      
          + Drop "per hour" and "employer provided" -> replace and append as features 
      
          + Get min, max and avg
      
          + Convert hourly wages into salary
      
      * Company name parsing:
      
          + Company name only - get rid of everything else in that field 
      
      * State field 
      
      * Age of company 
      
      * Get job description length and put it in as a feature
      
      * Parsing of job description (python, R, etc.) 
      
      * Parsing job titles:
      
          + Simplifying job titles:
      
              + Using a list of keywords, simplify the titles so they can be
      
              + grouped together. i.e titles like 
      
              + "Research Data Scientist", "Chemistry Data Scientist",
      
              + "Buisness Data Scientist" etc. will all have the simplified title
      
              + of "Data Scientist"
      
          + Parsing for seniority:
      
              + Parse the titles for keywords to indicate seniority level of the position
      
      * Convert catagorical/object data to numeric
      
          + First convert all object data to catagorical
      
      * Drop rows where Ratings = -1
    """)
    
    show_code = st.sidebar.checkbox("Show Data Cleaning Code", False)
    
    if show_code:
        st.markdown("""
        ```
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
        current_year = 2020
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
        
        
        'Converting catagorical/object data to numeric'
        
        #First I have to convert all the objects to catagorical data
        df = df.apply(lambda x: x.astype('category') if x.dtype=='object' else x)
        
        #Convert revenue to numeric, in order
        df['revenue_num'] = df.Revenue.apply(convert_revenue_no_unknown).astype('int64')
        df['revenue_num_unknown']= df.Revenue.apply(convert_revenue).astype('int64')
        
        #Convert size to numeric, in order
        df['size_num'] = df.Size.apply(convert_size_no_unknown).astype('int64')
        df['size_num_unknown']= df.Size.apply(convert_size).astype('int64')
        
        #Convert all desired catagorical data to numeric
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # Assigning numerical values and storing in another column
        column_names = ["simplified_title",'seniority','location_state',
                        'Type of ownership', 'Sector']
        column_names_num = ["simplified_title_num",'seniority_num',
                            'location_state_num','type_of_ownership_num',
                            'sector_num']
        # Assigning numerical values and storing in another column
        for i in range(len(column_names)):
            df[column_names_num[i]] = labelencoder.fit_transform(df[column_names[i]])
        
        df
    
        # generate binary values using get_dummies
        dum_df = pd.get_dummies(df, columns=["simplified_title", 
                                             'seniority','location_state',
                                             'Type of ownership', 'Sector'])
        # merge with main df on key values
        df_merge = df.merge(dum_df, how='left')
        
        #Drop rows where Rating==-1
        df= df[df['Rating'] != -1]
                    
        ```         
                    """)
    else: st.markdown(""" _Click "Show Data Cleaning Code" if you'd like to see it_ """)
    
    st.markdown("""
    Here are some of the more interesting things I've found during my EDA: """)
    
    show_EDA = st.sidebar.checkbox("Show EDA", True)
    if show_EDA:
        eda()
    
    
    st.markdown("""
    ## Phase 05: Modeling 
    
    In class, we had only learnt about 4 algorithms: Linear Regression, Logistic regression, K-NN classification and K-Means clustering. Given that my goal was to predict a salary estimate, a continuous variable, logistic regression and K-NN classification were out since they could only be used to classify categorical variables. That left me with K-Means and LM. K-Means *could* be useful and *might* give me some interesting clusters, but it wouldn't help me predict my target variable. That left me with just LM. With that decided, I started researching how to implement Linear Regression in python. While doing so, I found that python had *two* packages that could be used to implement this: the statsmodel and the sklearns package. I could not really discern the difference between the two or discern which was better, so I implemented them both! The statsmodel LM had an accuracy score (measured by adjusted r-squared) of roughly 74% while the sklearns model had an accuracy of about 79%. While looking at the summary for the statsmodel LM, I noticed that a lot of the variables had P>|t| values that where greater than 0.05, meaning that they were not significant. I also noticed that a lot of the variables were co-related to each other (i.e. Sector and Industry) and were probably also skewing the model. As I was searching for the best ways to remedy these issues, I came across [this article](https://dataaspirant.com/lasso-regression/#t-1606404715785) which talks about Lasso regression. Lasso regression seemed to do exactly what I was looking for, so I implemented that as well! The accuracy of my Lasso regression is roughly at 80%.
    
    ## Phase 06: Building a data product
    
    Now that I had my models, it was time to build my data product! I decided I wanted my data product to be a web app, and I built it using streamlit. Of the four models I had made, I chose to forgoe the OLS LM because it was less accurate and the K-Means beacuse I was unable to represnent the clusters in any meaningful fashion. 
    
    ## Phase 07: Documenting & Deploying
    
    Depending on how you are viewing it, I have documenting this project using either Rmd or streamlit markdown. I have deployed it via Heroku.
    
    ## Resources:
    
    https://github.com/arapfaik/scraping-glassdoor-selenium/blob/master/.ipynb_checkpoints/glassdoor%20scraping-checkpoint.ipynb
    https://github.com/SergeyPirogov/webdriver_manager
    
    	
    """)


st.title('Salary Estimator App')

def main():
    choose_model = st.sidebar.selectbox("Choose the ML Model",
    		["SKLearns Linear Regression", "Lasso Regression"])
    
    show_data = st.sidebar.checkbox("Show Raw Data", False)
    
    if show_data:
        st.subheader("Showing raw data ---->>>")
        st.write(data)
    
    st.write("The current model being used is: ", choose_model)
    
    if(choose_model == "SKLearns Linear Regression"):
        use_model(sklearns_lm)
        
    if(choose_model == "Lasso Regression"):
        use_model(lasso_regression)

def input_user_data(template):
    titles = ['Data Scientist', 'Other', 'Data Analyst', 'Other Scientist',
              'Research Scientist', 'Data Engineer', 'Other Analyst',
              'Machine Learning Engineer', 'Instructor', 'Research Analyst']
    states = [' DC', ' MD', ' CA', ' WA', ' TN', ' TX', ' WI', ' NY', ' ID',
              ' OH', ' FL', ' NC', ' VA', ' MO', ' MI', ' MA', ' MN', ' IL',
              ' CT', ' CO', ' PA', ' OR', ' NJ', ' NH', ' GA', ' NV', ' AZ',
              ' WY', ' UT', ' SC', ' AR', ' AL', ' IN', ' RI']
    states.sort()
    titles.sort()
    
    title = st.selectbox("What is the Job Title?", titles)
    seniority = st.selectbox('What is the seniority leve of the position?',
                             ['Senior','Mid-level','Junior','Not Specified'])
    state = st.selectbox("What State is the job in?", states)

    user_data = interpret_base_input(title, seniority, state, template)
    
    if st.checkbox("Set advanced inputs"):
        user_data= get_advanced_inputs(user_data)
    
    return user_data

def get_advanced_inputs(user_data):
    rating = st.number_input("How is this company rated on a scale of 1-5?",
                                 min_value=1.0, max_value=5.0, 
                                 value=3.5, step=0.1)
    user_data.Rating = rating
    
    contains = st.multiselect("Does the job description require:",
                              ["Python", "R", 
                               "Big Data(AWS, Hadoop, Spark, etc.)", 
                               "Masters Degree"])
    user_data= interpret_contains(contains, user_data)
    
    pay = st.selectbox("How will you be paid?", ['Salary','Hourly'])
    if pay=='Hourly': user_data.hourly=1
    
    age = st.number_input("How old is the company?", 
                          min_value=0, value=36, step=1)
    user_data.age = age
    
    size = st.selectbox("How big is this company?",
                        ['1 to 50 Employees', '51 to 200 Employees', 
                         '201 to 500 Employees',
                         '501 to 1000 Employees','1001 to 5000 Employees',
                         '5001 to 10000 Employees','10000+ Employees'])
    user_data= interpret_size(size,user_data)
    
    rev = st.selectbox("What is the company's revenue?",
                       ['Less than $1 million (USD)','$1 to $5 million (USD)',
                        '$5 to $10 million (USD)','$10 to $25 million (USD)',
                        '$25 to $50 million (USD)','$50 to $100 million (USD)',
                        '$100 to $500 million (USD)','$500 million to $1 billion (USD)',
                        '$1 to $2 billion (USD)','$2 to $5 billion (USD)',
                        '$5 to $10 billion (USD)','$10+ billion (USD)'])
    user_data= interpret_revenue(rev,user_data)
    
    types = ['Company - Public', 'Company - Private', 'Nonprofit Organization',
       'Subsidiary or Business Segment', 'Hospital',
       'College / University', 'Government', 'Other Organization',
       'Private Practice / Firm', 'Contract']
    types.sort()
    ownership_type = st.selectbox("What is the Type of Ownership?",types)
    user_data= interpret_type(ownership_type,user_data)
    
    sectors = ['Information Technology', 'Business Services', 'Manufacturing',
       'Transportation & Logistics', 'Health Care', 'Insurance',
       'Finance', 'Real Estate', 'Aerospace & Defense',
       'Telecommunications', 'Government', 'Media', 'Education',
       'Oil, Gas, Energy & Utilities', 'Biotech & Pharmaceuticals',
       'Non-Profit', 'Travel & Tourism', 'Retail',
       'Construction, Repair & Maintenance', 'Accounting & Legal',
       'Arts, Entertainment & Recreation', 'Agriculture & Forestry',
       'Mining & Metals']
    sectors.sort()
    sect = st.selectbox("What Sector is the job in?", sectors)
    user_data=interpret_sector(sect,user_data)
    
    return user_data

#Takes user inputs and correctly fills in  the template with it
def interpret_base_input(title, seniority, state, template):
    user_data = template.copy()
    for column in template.columns:
        if title in column:
            user_data[column] = 1
        if seniority in column:
            user_data[column] = 1
        if state in column:
            user_data[column] = 1
    return user_data

def interpret_contains(contains, template):
    user_data = template
    
    if any(x=="Python" for x in contains): user_data.contains_python=1
    if any(x=="R" for x in contains): user_data.contains_R=1
    if any(x=="Big Data(AWS, Hadoop, Spark, etc.)" for x in contains): 
        user_data.contains_big_data=1
    if any(x=="Masters Degree" for x in contains): user_data.contains_MS=1
    return user_data

def interpret_size(size,user_data):
    if size=='1 to 50 Employees':
        user_data.size_num= 1
    elif size=='51 to 200 Employees':
        user_data.size_num= 2
    elif size=='201 to 500 Employees':
        user_data.size_num=3
    elif size=='501 to 1000 Employees':
        user_data.size_num=4
    elif size=='1001 to 5000 Employees':
        user_data.size_num=5
    elif size=='5001 to 10000 Employees':
        user_data.size_num=6
    elif size=='10000+ Employees':
        user_data.size_num=7
    return user_data

def interpret_revenue(revenue,user_data):
    if revenue=='Less than $1 million (USD)':
         user_data.revenue_num=1
    elif revenue=='$1 to $5 million (USD)':
        user_data.revenue_num=2
    elif revenue=='$5 to $10 million (USD)':
        user_data.revenue_num=3
    elif revenue=='$10 to $25 million (USD)':
        user_data.revenue_num=4
    elif revenue=='$25 to $50 million (USD)':
        user_data.revenue_num=5
    elif revenue=='$50 to $100 million (USD)':
        user_data.revenue_num=6
    elif revenue=='$100 to $500 million (USD)':
        user_data.revenue_num=7
    elif revenue=='$500 million to $1 billion (USD)':
        user_data.revenue_num=8
    elif revenue=='$1 to $2 billion (USD)':
        user_data.revenue_num=9
    elif revenue=='$2 to $5 billion (USD)':
        user_data.revenue_num=10
    elif revenue=='$5 to $10 billion (USD)':
        user_data.revenue_num=11
    elif revenue=='$10+ billion (USD)':
        user_data.revenue_num=12
    return user_data

def interpret_type(ownership_type, user_data):
    for column in user_data.columns:
        if ownership_type in column:
            user_data[column] = 1
    return user_data

def interpret_sector(sect, user_data):
    for column in user_data.columns:
        if sect in column:
            user_data[column] = 1
    return user_data

# Training the SKLearns Lm model
@st.cache
def sklearns_lm(x_train, x_test, y_train, y_test):
    #Using the sklearns library
    # fit a model
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    #y_pred = lm.predict(x_test)
    accuracy = (model.score(x_test, y_test)) * 100
    return accuracy, lm

@st.cache
def lasso_regression(x_train, x_test, y_train, y_test):
    ## define model evaluation method
    cross_validation = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    ## define model
    lasso_model = LassoCV(alphas=arange(0, 1, 0.02), cv=cross_validation , n_jobs=-1)
    
    ## fit model
    lasso_model.fit(x_train, y_train)
    
    pred_train_lasso= lasso_model.predict(x_train)
    pred_test_lasso= lasso_model .predict(x_test)
    accuracy = (r2_score(y_test, pred_test_lasso)) * 100
    
    return accuracy, lasso_model

def use_model(model):
    accuracy, model = model(x_train, x_test, y_train, y_test)
    st.text('Accuracy of the model is: ')
    st.write(accuracy, '%')
    try:
        if(st.checkbox('Predict with own data', True)):
            user_data = input_user_data(template)
            if(st.sidebar.checkbox("Show User Input")):
                st.write(user_data)
            if st.button("Predict"):
                prediction = int(model.predict(user_data) *1000)
                prediction = locale.format("%d", prediction, grouping=True)
                st.write("You should get paid around: $", prediction, "per year")
            
            return
    except Exception as err:
        st.write(err)
        pass

main()