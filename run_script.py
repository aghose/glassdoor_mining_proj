# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:37:31 2020

@author: Akash Ghose
"""

import glassdoor_scraper as gs
import pandas as pd

keyword = 'Data Scientist'

chromedriver_path = "C:/Users/asgho/github/glassdoor_mining_proj/chromedriver"
sleep_time = 7
num_jobs = 5

#This line will open a new chrome window and start the scraping.
df = gs.get_jobs(keyword, num_jobs, False, chromedriver_path, sleep_time)

df