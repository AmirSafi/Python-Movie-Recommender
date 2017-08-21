#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Movie Recommender 
Collaborative Filtering 
using MovieLens 100k data set 
Created on Wed Aug  9 21:31:35 2017
@author: Amir
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
#These libraries are used in the 
#exploratory data analysis method
from math import sqrt, pi 
import matplotlib.pyplot as plt
import scipy.stats

#Based on 100K data set
numUsers = 943
numItems = 1682
numRatings = 100000

ratingColumns = ['user id', 'item id', 'rating', 'timestamp']
#store maping between movie ID and movie title 
movieTitleMap= {}
#item correlation matrix, stored here becuase the correlation between items 
#doesn't change that often as compared to user-user correlation 
itemCorrMatrix = np.zeros((numItems+1 , numItems+1))
#store genre name and ID
genre_dict = {}


def importSQLdatabase():
    """Import data from sqlite3 database
    @Return df Pandas Dataframe with all the data and labeled columns"""
    connection = sqlite3.connect('../database/100k/sqlite/MovieLens100k.db')
    sqlDB = connection.execute('SELECT * FROM rating')
    data = [row for row in sqlDB]
    df = pd.DataFrame(data , columns = ratingColumns)
    return df


def importAllRatingData():
    """Import data into pandas dataframe 
    for now I'm ignoring the timestamp data. 
    could use the date information to recommend latest movies
    will implement later on
    @Return: Pandas Dataframe with all the data and labeled columns"""
    Location = '../datasets/100k/ml-100k/ml-100k/u.data'
    df = pd.read_csv(Location, sep = '\t', header = None, names = ratingColumns)
    return df


def importTestData(fileName):
    """80%/20% splits of the u data into training and test data.
    @Param: fileName Name of the base or test set
    @Return: df Pandas Dataframe with test data and labeled columns"""
    Location = '../datasets/100k/ml-100k/ml-100k/' + fileName 
    df = pd.read_csv(Location, sep = '\t', header = None, names = ratingColumns)
    return df


def importGenre():
    """Import list of genre
    @Return: df Pandas Dataframe with movie genre and ID"""
    Location = '../datasets/100k/ml-100k/ml-100k/u.genre'
    df = pd.read_csv(Location, sep = '|', header = None, names = ['genre', 'id'])
    for x in df.values:
        genre = x[0]
        genreID = x[1]
        genre_dict[genre] = genreID
    return df

def importMovieData():
    """Import entire movie dataset with all movie information
    @Return df Pandas dataframe of the movie data with labeled columns"""
    Location = '../datasets/100k/ml-100k/ml-100k/u.item'
    itemInfoList = ['movie id' ,'movie title' , 'release date' , 'video release date', \
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' , \
              'Children\'s' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,\
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' , \
              'Thriller' , 'War' , 'Western' ]
    df = pd.read_csv(Location , sep = '|' , header = None, names = itemInfoList)
    #create a map between movie ID and movie title 
    for movie in df.values:
        movieID = movie[0]
        movieTitle = movie[1]
        movieTitleMap[movieID] = movieTitle
    return df

def timeSpan(data):
    """Returns the time span of the rating data as a string 
    @Param: data Pandas Dataframe
    @Return: span A string representing the time span of movies in the system""" 
    max = data.apply(np.max)['timestamp']
    min = data.apply(np.min)['timestamp']
    maxStr = datetime.fromtimestamp(max).strftime('%m/%d/%Y')
    minStr = datetime.fromtimestamp(min).strftime('%m/%d/%Y')
    span = 'From : ' + minStr + ' To: ' + maxStr
    return span


def userItemMatrix(data , nUsers , nItems):
    """Create user-item matrix ratings matrix  
    Note: This matrix is quite sparse. We have just 100,000 ratings. 
    Only 6% (100,000/ 943*1682) of the matrix is filled   
    @Param: data Pandas dataframe containing ratings 
    @Param: nUsers Number of users 
    @Param: nItems Number of Items
    @Return matrix A user-item matrix (2D numpy Array)""" 
    matrix = np.zeros((nUsers + 1 , nItems + 1))
    for x in data.values:
        user = x[0]
        item = x[1]
        rating = x[2]
        matrix[user][item] = rating    
    return matrix

   
def mostPopularMovie(matrix, n=5):
    """Simple Popularity Model 
    @Param: matrix User-Item matrix
    @Param: n Size of the recommendation list  
    @Return: most popular movies based on average ratings""" 
    #Precondition Check
    if n > numItems:
        return None
    #store movie ID , rating and rating count 
    movies = {}
    #initialize the dictionary 
    for x in range(numItems + 1):
        movies[x] = {'totRating':0 , 'count':0 , 'avgRating':0}
    #item ID counter 
    i = 0 
    #Transpose matrix to iterate over the items 
    for item in matrix.T:
        i += 1
        for rating in item:
            if rating != 0:
                movies[i-1]['totRating'] += rating
                movies[i-1]['count'] += 1           
    for key, value in movies.iteritems():
        #If movie has at least one rating update the average rating 
        if movies[key]['count'] >= 1:
            avgRating = movies[key]['totRating']/movies[key]['count']
            movies[key]['avgRating'] = avgRating                 
    #sort by the average rating and return top n movies         
    items = movies.items()
    items.sort(key = lambda item: (item[1], item[0]))
    #List comprehension to get tuples of movie ID and average rating
    sortedRatings = [(item[1]['avgRating'], item[0]) for item in items]
    topN = sortedRatings[-n:]
    topList = []
    for x in topN:
        topList.append(tuple((x[0],movieTitleMap[x[1]])))
    orderedList = list(reversed(topList))    
    return orderedList

  
def userCFrecommender(matrix, userID ,k = 10 , n = 5):
    """Nearest-neighbor User-User Collaborative filter 
    @Param: matrix User-Item matrix
    @Param: k Number of nearest neighbors 
    @Param: n Number of movies to recommend 
    @Return list A list of top k items for userID using collabrotive filtering"""  
    numUsers = matrix.shape[0] - 1
    numItems = matrix.shape[1] - 1
    #Precondtion Check 
    if k > numUsers or userID > numUsers:
        return None
    #Create a user1-to otherUser correlation vector
    userCorrVector = np.zeros((1 ,numUsers + 1))
    neighbors = []
    #find the k nearses neighbors to the user
    for user2 in range(1 , numUsers + 1):
        '''The pearson correlation method I implemented was very unefficient 
        I read SciPy documentaion and found a function called scipy.stats.personr
        Numpy also has a simillar method called numpy.corrcoef
        the numpy and my pearson correlation methods give slightly different correlation 
        coefficients. This is becuase I use a vector that is the length of the number of 
        movies 2 people have in common rather than a vector with length equal to number of items 
        The flaw with my method was that majority of people dont have rated movied in common 
        making their simillarity vector length 0. ''' 
        correlation = np.corrcoef(matrix[userID], matrix[user2])[0][1]
        userCorrVector[0][user2] = correlation
        #Exclude self correlation 
        if user2 != userID:
            neighbors.append(tuple((correlation, user2)))
        #userCorrVector[0][user2] = np.corrcoef(matrix[userID], matrix[user2])[0][1]
    neighbors.sort()
    kNearest = neighbors[-k:]
    #Find the top movies that the neighbors have rated 
    topMovies_dict = {}
    for x in range(1, numItems +1):
        #simSum = simillarity sum between a the person(userID) and his/her neighbor
        topMovies_dict[x] = {'total':0 , 'simSum':0 ,'rating':0}   
    neighborIDs = [x[1] for x in kNearest]
    '''Transverse all the neighbors and calculate the estimate rating for a movie by 
    multiplying the neighbors rating with user-neighbor simillarity'''
    for neighbor in neighborIDs:
        for movie in range(1 , numItems +1 ):    
            topMovies_dict[movie]['total'] += matrix[neighbor][movie] * userCorrVector[0][neighbor]
            topMovies_dict[movie]['simSum'] += userCorrVector[0][neighbor]          
    #calculate the adjusted ratings        
    for x in range(1, numItems + 1):
       topMovies_dict[x]['rating'] = topMovies_dict[x]['total']/topMovies_dict[x]['simSum']
     
    #sort by the average rating and return top n movies         
    items = topMovies_dict.items()
    #print items
    items.sort(key = lambda item: (item[1], item[0]))
    #List comprehension to get tuples of movie ID and average rating
    sortedRatings = [(item[1]['rating'], item[0]) for item in items if matrix[userID][item[0]] ==  0]
    topN = sortedRatings[-n:]
    topList = []
    for x in topN:
        topList.append(tuple((x[0],movieTitleMap[x[1]])))
    orderedList = list(reversed(topList)) 
    return orderedList

  
def itemCF(matrix, itemID, k = 5):
    """Nearest-neighbor item-item Collaborative filter 
    @Param: matrix User-Item matrix
    @Param: k Number of nearest neighbors
    @Return orderedList A list of k simillar items to the provided item""" 
    numItems = matrix.shape[1] - 1
    #Precondtion Check 
    if k > numItems:
        return None 
    #transpose the matrix 
    matrix = matrix.T
    #itemCorrVector = np.zeros((1 ,numItems + 1))
    neighbors = []
    #find the k nearses neighbors to the user
    for item2 in range(1 , numItems):
        correlation = np.corrcoef(matrix[itemID], matrix[item2])[0][1]
        #if item2 != itemID:
        neighbors.append(tuple((correlation, item2)))
    neighbors.sort()
    kNearest = neighbors[-k:]
    topList = []
    for x in kNearest:
        topList.append(tuple((x[0],movieTitleMap[x[1]])))
    orderedList = list(reversed(topList)) 
    return orderedList

def svd(matrix):
    """Matrix Factorization , Singular Value Decomposition
    R = P*s*Qt
    P = m x n ratings matrix 
    s = k x k diagonal feature weight matrix (singular values)
    Q = n x k item-feature relevance matrix , Qt = Q transpose
    Prediction Rule r-ui 
    
    
    @Param: matrix A User-Item matrix 
    @Return:   """
    #normalize the matrix by subtracting the mean off 
    #mean rating for each user
    mean = np.mean(matrix,1)
    #Transpose the row vector
    meanT = np.asarray([(mean)]).T
    normR = matrix - meanT
    #print norm[7][479]
    #print matrix[7][479]
    P, s , Qt = np.linalg.svd(normR , full_matrices = False)
    S = np.diag(s)
    
    print P[1].shape
    print S[1].shape
    print Qt[1].shape
    #v = np.dot(P[1], np.dot(S[1],Qt[1].T))
    #print np.dot(P, np.dot(S,Qt))
    #print np.dot(s , np.identity(s.size))
    #zz = np.dot(s , np.identity(s.size))
    #print P.shape
    #print s.shape
    #print Qt.shape   
    #print np.dot(np.dot(P,s),Qt)
    #print P
    return None


def baseline(matrix, userID):
    """Estimate the baseline rating of an item the user hasn't seen 
    as their average rating of all the movies they have rated.
    @Param: matrix A User-Item matrix
    @Param: userID The user for which to estimate baseline rating
    @Return: avgRating The average rating of the user"""
    #precondition check
    if userID > numUsers or userID < 0:
        return None
    total = 0
    numRating = 0
    for x in matrix[userID]:
        if x > 0:
            total += x
            numRating += 1
    if numRating == 0:
        return None        
    avgRating = total/numRating   
    return avgRating


def pearsonCorrUsers(matrix, user1, user2):
    """Similarity function: Returns similarity measure of two objects (user or items)
    Caclualted by the pearson correlation coefficient    
    Time complexity of this algorithim is O(n^2) where n is the number of users 
    This is computationaly expensive and should be run only when necessary 
    @Param: matrix A User-Item matrix
    @Param: user1 The first user 
    @Param: user2 The secodn user
    @Return: r The person correlation coefficient between user1 and user2"""      
    #Precondition Check 
    if user1 > numUsers or user2 > numUsers:
        return None 
    #sample size = numer of elements in common between users 
    n = 0
    x = []
    y = []
    xy = []
    x_sq = []
    y_sq = []
    for item in range(0, numItems):
        if matrix[user1][item] != 0 and matrix[user2][item] !=0:
            x.append(matrix[user1][item])
            y.append(matrix[user2][item])
            xy.append(matrix[user1][item] * matrix[user2][item])
            x_sq.append(matrix[user1][item] ** 2)
            y_sq.append(matrix[user2][item] ** 2)
            n +=1
    if n == 0:
        return 0        
    sum_x = sum(x)
    sum_y =  sum(y)      
    sum_xy = sum(xy)
    sum_x_sq = sum(x_sq) 
    sum_y_sq = sum(y_sq)  
    num = n * sum_xy - (sum_x * sum_y)
    den = sqrt(((n*sum_x_sq - (sum_x**2)) * (n*sum_y_sq - (sum_y**2))))
    r = num/den 
    #print 'user ' + str(user1) + ' and user ' + str(user2) + ' have ' + str(n) + ' items in common ' + 'correlation = ' + str(r)
    return r


def testMovieObject(movieData):
    """Practice OOP in python. Create Movie Objects 
    @Param: movieData A Pandas Datafame with movie data and labeled columns"""
    movie_Dict = {}
    for movie in range(1 , movieData.shape[0]):
        movieID = movieData.loc[0,['movie id']]
        movieTitle = movieData.loc[0,['movie title']]
        releaseDate = movieData.loc[0,['release date']]
        IMDbURL = movieData.loc[0,['IMDb URL']]
        #Create Movie objects and add to dictionary 
        movie_Dict["movie" + str(movie)] = Movie(movieID, movieTitle, releaseDate , IMDbURL)
        
    #print movie_Dict.get('movie1').toString()
    return None 
   

if __name__ == '__main__':
    importSQLdatabase()
    ratingData = importAllRatingData()
    movieData = importMovieData()
    importTestData('ua.base')
    importGenre()
    timeSpan(ratingData)
    matrix = userItemMatrix(ratingData , numUsers, numItems)
    topList = mostPopularMovie(matrix)
    #print topList
    userCFrecommender(matrix , 66)
    itemCF(matrix, 538)
    baseline(matrix, 8)
    pearsonCorrUsers(matrix, 60,30)
    testMovieObject(movieData)
    svd(matrix)

