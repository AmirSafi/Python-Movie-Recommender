# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:23:43 2017

@author: Amir
"""
#import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equals, assert_almost_equals
#add python path at runtime 
import sys
sys.path.insert(0,'../models')
import recommender


#import a small subset of the data and test the various methods in recommender.py file 
#Here I will make a matrix of 10 users and 10 items with some ratings 
nUsers = 10
nItems = 10
d = {'user id' : [1] , 'item id':[2] , 'rating':[5], 'timestamp': [874965758]}
pSet = pd.DataFrame(d, columns = ['user id' , 'item id' , 'rating', 'timestamp'])

trainSet1 = importTestData('u1.base')

matrix = userItemMatrix(pSet , nUsers , nItems)


print matrix
print matrix.shape


print pSet
print trainSet1.iloc[0:2]

def testUserItemMatrixDim():
    assert_equals(1,1)
