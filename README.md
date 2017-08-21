Python-Movie-Recommender
Author: Amir Ali 
==============================================
Built using Python 2.7.13 |
with Python Anaconda 4.4.0 (64-bit) IDE 
Required Python Modules = [pandas, numpy, scipy, sklearn, sqlite3, matplotlib ]
-------------------------------------------------------------------------------
The data used is provided by GroupLens Research Project at the University of Minnesota.
Downloaded data from 
https://grouplens.org/datasets/movielens/

There are 3 data sets used:
*  100k - Consists of 100,000 ratings (1-5) from 943 users on 1682 movies
*  1m   - Consists of 1,000,209 anonymous ratings of approximately 3,900 movies 
          made by 6,040 MovieLens users
*  20m  - Consists of 20000263 ratings and 465564 tag applications across 27278 movies
**See the README.txt file in each data set folder for more details. 
-------------------------------------------------------------------

There are 3 python recommenders:
*  recommender.py - Uses the 100k data set. Baseline recommender using collaborative filtering.

Notes:
- There are a lot of design choices when making a recommendation system.
  I have started the Coursera Specialization in Recommender Systems to get a better 
  understanding of the pros and cons each design choice. 
  
-  

Journal

Day1: 
-import data into python. Started with the 100k dataset.
-learned pandas and numpy libraries. Read documentations. 
-created a user item matrix using pandas and numpy libraries.   
-implemented most popular movie recommender 

Day 2: 
-Implemented and tested pearson correlation measure method 
-Implemented collabrotive filtering both user and item based 

Day 3:
-Added metrics, evaluation and testing 
-will add accuracy and error measures: MAE, RMSE and MSE
-Setting up the evaluation methods now is necessary for tuning and optimizing the recommender. 

Day 4:
-Learned/Reviewed SQLite commands 
-Created a database and read the data using python 
-Added to test cases for the first recommender. 
-Added proper python method comments. 
-Tested item and user CF methods with a hand made small data set. 

Day 5:
-Added SVD method. Normalized the matrix before factorization and calcualted the RMSE 
-based on the test set. 
-Transitioned to the 20 milllion ratings dataset. 
-Will implement stochastic gradient descent to estimate the SVD for larger matrices. 
