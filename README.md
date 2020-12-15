## Technologies used:

Python, pandas, numpy, seaborn, matplotlib, streamlit, Rmarkdown, KnitR and Heroku

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

## Phase 01: Project Planning

**Area of research:** Glassdoor Job Market.

**Title of project:** Glassdoor Data analysis and Salary estimator.

**Potential clients:** Students or people looking for jobs.

**Potential sponsors:** People who wished scrape Glassdoor and anaylize the job market for a given job.

**Objective:** Sometime back in October, I was on [Glassdoor](https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword="+'Data Scientist'+"&locT=&locId=0&jobType=&context=Jobs&sc.keyword="+'Data Scientist'+"&dropdown=0) looking at job postings for Data Scientists. As I browsed through them, I had several questions. I wondered how many of those jobs required Python vs R. Or how many wanted you to have a masters degree. I wondered which States paid you the most. I wondered if there was difference in average pay amongst the different job titles I noticed. I also noticed that some of the job listings did not have any Glassdoor estimates for salaries. I wondered if it might be possible for me to be able to *predict* some of these estimates. It seemed like a reach at that time, given my level of knowledge and experience, but I thought, why not try. 

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

Here are some interesting things I've found during my EDA:

## Phase 05: Modeling 

In class, we had only learnt about 4 algorithms: Linear Regression, Logistic regression, K-NN classification and K-Means clustering. Given that my goal was to predict a salary estimate, a continuous variable, logistic regression and K-NN classification were out since they could only be used to classify categorical variables. That left me with K-Means and LM. K-Means *could* be useful and *might* give me some interesting clusters, but it wouldn't help me predict my target variable. That left me with just LM. With that decided, I started researching how to implement Linear Regression in python. While doing so, I found that python had *two* packages that could be used to implement this: the statsmodel and the sklearns package. I could not really discern the difference between the two or discern which was better, so I implemented them both! The statsmodel LM had an accuracy score (measured by adjusted r-squared) of roughly 74% while the sklearns model had an accuracy of about 79%. While looking at the summary for the statsmodel LM, I noticed that a lot of the variables had P>|t| values that where greater than 0.05, meaning that they were not significant. I also noticed that a lot of the variables were co-related to each other (i.e. Sector and Industry) and were probably also skewing the model. As I was searching for the best ways to remedy these issues, I came across [this article](https://dataaspirant.com/lasso-regression/#t-1606404715785) which talks about Lasso regression. Lasso regression seemed to do exactly what I was looking for, so I implemented that as well! The accuracy of my Lasso regression is roughly at 80%.

## Phase 06: Building a data product

Now that I had my models, it was time to build my data product! I decided I wanted my data product to be a web app, and I built it using streamlit.

## Phase 07: Documenting & Deploying

I am documenting this project using Rmd. I have deployed it via Heroku.

## Resources:

https://github.com/arapfaik/scraping-glassdoor-selenium/blob/master/.ipynb_checkpoints/glassdoor%20scraping-checkpoint.ipynb
https://github.com/SergeyPirogov/webdriver_manager
