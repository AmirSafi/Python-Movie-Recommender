#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Movie Recommender 

Will implement a variety of recommenders 
-Start with non-personalized and content based recommenders 
-Then implement nearest neighbor collaborative filtering 
-Finally use the large data set (20 million ratings) and
implement matrix factorization and advanced techniques for large matrices  


Movie Lens data set 

Created on Wed Aug  9 21:31:35 2017

@author: Amir
"""
import pandas as pd
import numpy as np

    
# import data into pandas dataframe 
# for now I'm ignoring the timestamp data. 
# could use the date information to recommend latest movies
# will implement later on
def importRatingData():
    uLocation = '../datasets/100k/ml-100k/ml-100k/u.data'
    df = pd.read_csv(uLocation, sep = '\t', header = None, names = ['user id', 'item id', 'rating', 'timestamp'])
    return df

# import movie data. Use to get movie name from the movie id.
# The movie features will be used to represent each movie and user in the same 
# dimensions to calculate similarites (user-user , user-item and item-item)
# Will also need to implement a similarity function   
def importMovieData():
    iLocation = '../datasets/100k/ml-100k/ml-100k/u_item.item'
    itemInfoList = ['movie id' ,'movie title' , 'release date' , 'video release date', \
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' , \
              'Children\'s' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,\
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' , \
              'Thriller' , 'War' , 'Western' ]
    # store each movie title and ID in python dictionary           
    movies = {}    
    for line in open(iLocation):
        (id,title)=line.split('|')[0:2]
        movies[id]=title
    return movies
    

# Create and return a user-item matrix with ratings    
# Note: This matrix is quite sparse. We have just 100,000 ratings. 
# Only 6% (100,000/ 943*1682) of the matrix is filled    
def userItemMatrix(data):
    #get and sort all unique user and item IDs
    userIDs = data['user id'].unique()
    userIDs.sort()
    itemIDs = data['item id'].unique()
    itemIDs.sort()
    emptyMatrix = np.empty([943,1682])
    matrix = pd.DataFrame(emptyMatrix, columns = itemIDs , index = userIDs)
    #add the ratings to the matrix 
    for index, row in data.iterrows():
        #get matrix indices and ratings as integers 
        rowID = row['user id'] + 0
        colID = row['item id'] + 0
        rating = row['rating'] + 0
        ##fill in the ratings 
        matrix[rowID][colID] = rating
    return matrix

# Non-personalized recommender
def recommendMostPopular(matrix , n = 5):
    movieScore= {}
    #iterate through the matrix and calculate normalized scores for each movie
    #normalize the score by the number or ratings
    for index, row in matrix.iterrows():
        if (row['item id'] + 0) in movieScore:
            movieScore[row['item id'] + 0] += row['rating'] + 0
        else:
            movieScore[row['item id'] + 0] = 0 
    #normalize each score
    #reverse key value pair so we can rank by the normalized score of each movie
    topMovies = []
    #return top k most popular movies 
    return topMovies[0:n]   
    
    
if __name__ == '__main__':
    #print(__file__)
    rating = importRatingData()
    #movies = importMovieData()
    matrix = userItemMatrix(rating)
    recommendMostPopular(matrix)
        