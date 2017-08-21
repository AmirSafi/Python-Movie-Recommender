#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:02:49 2017

@author: Amir
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from datetime import datetime
from math import sqrt, pi , pow
import matplotlib.pyplot as plt
import scipy.stats

#Based on 20m data set
numUsers = 138493
numItems = 131262
numRatings = 20000264
ratingColumns = ['user id', 'item id', 'rating', 'timestamp']

def importSQLdatabase():
    """Import data from sqlite3 database
    @Return df Pandas Dataframe with all the data and labeled columns"""
    connection = sqlite3.connect('../database/20m/sqlite/MovieLens20m.db')
    sqlDB = connection.execute('SELECT * FROM rating')
    data = [row for row in sqlDB]
    df = pd.DataFrame(data , columns = ratingColumns)
    print df.size
    print df.shape
    return df

def userItemMatrix(data , nUsers , nItems):
    """Create user-item matrix ratings matrix  
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

def svd(matrix , test, k):
    """Matrix Factorization , Singular Value Decomposition
    R = P*s*Qt
    P = m x n ratings matrix 
    s = k x k diagonal feature weight matrix (singular values)
    Q = n x k item-feature relevance matrix , Qt = Q transpose
    Prediction Rule r-ui = users u's rating for item i. 
    r-ui = sum over f features of puf*sigmaf*qif
    @Param: matrix A User-Item matrix 
    @Param: test A test set 
    @Param: k The number of features to use
    @Return: rmse Root Mean Squared Error  """
    #normalize the matrix by subtracting the mean off mean rating for each user
    mean = np.mean(matrix,1)
    #Transpose the row vector
    meanT = np.asarray([(mean)]).T
    normR = matrix - meanT
    P, s , Qt = np.linalg.svd(normR , full_matrices = False)
    S = np.diag(s)
    n = test.size 
    sumError = 0 
    for x in test.values:
        user = x[0]
        item = x[1]
        rating = x[2]
        rui = 0
        bui = baseline(matrix, user)
        #Estimate based on k latent features 
        for f in range(0, k):
            rui += P[user][f]*S[f][f]*Qt[f][item]
        #Add back the baseline predictor
        rui += bui
        if rui > 5:
            rui = 5
        if rui < 1:
            rui = 1
        deltaSQ = pow(rating - rui , 2) 
        sumError += deltaSQ
        #print 'predicted rating ' , rui , 'actual rating ', str(rating)
    rmse =  sqrt(sumError/n)   
    print 'RMSE:' , rmse
    return rmse

if __name__ == '__main__':
    data = importSQLdatabase()
    train , test = train_test_split(data, test_size = 0.2)
    #trainMatrix = userItemMatrix(train, numUsers, numItems)
    #svd(trainMatrix, test , 100)
    