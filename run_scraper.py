# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:37:31 2020

@author: Akash Ghose
"""

import glassdoor_scraper as gs
#import pandas as pd

keyword = 'Data Scientist'

#Number of seconds you need the program to sleep in order to populate the 
#glassdoor page. Increase this if the program exits before populating the
#glassdoor page.
sleep_time = 7

#Number of job listing you want it to pull/scrape
num_jobs = 5

#This line will open a new chrome window and start the scraping.
df = gs.get_jobs(keyword, num_jobs, False, sleep_time)

#This will store the scraped data to a csv file
#df.to_csv('uncleaned_data.csv', index=False)
