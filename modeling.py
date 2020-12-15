# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 08:33:48 2020

@author: Akash Ghose 
"""

import pandas as pd
import statsmodels.api as sm # import statsmodels
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso #Import lasso regression
import numpy as np
from numpy import arange
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



data = pd.read_csv('cleaned_data.csv')

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
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

"LM models"
#Using statsmodel
x = sm.add_constant(x)
lm_model_sts = sm.OLS(y_train, x_train).fit() ## sm.OLS(output, input)
lm_predictions_sts = lm_model_sts.predict(x_test)

# Print out the statistics
lm_model_sts.summary()

#Using the sklearns library
# fit a model
lm = linear_model.LinearRegression()
lm_model_skl = lm.fit(x_train, y_train)
lm_predictions_skl = lm.predict(x_test)

#Plotting sklerans model
## The line / model
plt.scatter(y_test, lm_predictions_skl)
plt.xlabel('True Values')
plt.ylabel('Predictions')
#Printing accuracy score
skl_accuracy = lm_model_skl.score(x_test, y_test)
print('Accuracy Score: ', skl_accuracy)

"Lasso regression"
## define model evaluation method
cross_validation = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

## define model

lasso_model = LassoCV(alphas=arange(0, 1, 0.02), cv=cross_validation , n_jobs=-1)

## fit model
lasso_model.fit(x_train, y_train)
## summarize chosen configuration
print('alpha: %f' % lasso_model .alpha_)

pred_train_lasso= lasso_model.predict(x_train)
pred_test_lasso= lasso_model .predict(x_test)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso)))
print(r2_score(y_test, pred_test_lasso))

"K-means"

kmeans_df = normalize(df_merge)
kmeans_df = pd.DataFrame(kmeans_df, columns=df_merge.columns)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(kmeans_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#K=3
kmeans_3 = KMeans(n_clusters=3, init='k-means++', random_state=0)
pred_y_3 = kmeans_3.fit_predict(kmeans_df)

kmeans_clusters_df_3 = pd.DataFrame(kmeans_3.cluster_centers_)
kmeans_clusters_df_3.columns = df_merge.columns

#K=4
kmeans_4 = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=0)
pred_y_4 = kmeans_4.fit_predict(kmeans_df)

kmeans_clusters_df_4 = pd.DataFrame(kmeans_4.cluster_centers_)
kmeans_clusters_df_4.columns = df_merge.columns
