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

import numpy as np
sns.set_style("darkgrid")


st.title('Salary Estimator App')

st.markdown("""
	
""")
st.sidebar.title("Operations on the Dataset")

#st.subheader("Checkbox")
show_data = st.sidebar.checkbox("Show Raw Data", False)


@st.cache
def read_data():
    data = pd.read_csv('cleaned_data.csv')
    return data

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

def main():
    choose_model = st.sidebar.selectbox("Choose the ML Model",
    		["SKLearns LM"])
    
    #Bring in the data
    data = read_data()
    
    if show_data:
        st.subheader("Showing raw data ---->>>")
        st.write(data)
    
    x_train, x_test, y_train, y_test = preprocessing(data)
    
    template = make_template(x_train, data)
    
    if(choose_model == "SKLearns LM"):
        accuracy, lm = sklearns_lm(x_train, x_test, y_train, y_test)
        st.text('Accuracy of the model is: ')
        st.write(accuracy, '%')
        try:
            if(st.checkbox('Predict with own data')):
                user_data = input_user_data(template)
                if(st.sidebar.checkbox("Show User Input")):
                    st.write(user_data)
                if st.button("Predict"):
                    prediction = int(lm.predict(user_data) *1000)
                    prediction = locale.format("%d", prediction, grouping=True)
                    st.write("You should get paid around: $", prediction, "per year")
                
                return
        except Exception as err:
            st.write(err)
            pass
        
main()