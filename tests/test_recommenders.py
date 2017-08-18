# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:23:43 2017

@author: Amir
"""
import unittest 
#import pandas as pd
import numpy as np
#add python path at runtime to test the recommender 
import sys
sys.path.insert(0,'../models')
#import recommender

#import a small subset of the data and test the various methods in recommender.py file 
#Here I will make a matrix of 5 users and 7 items with some ratings 
nUsers = 7
nItems = 7
Location = '../tests/handMadeMatrix.data'
smallHandData = df = pd.read_csv(Location, sep = '\t', header = None, names = ratingColumns)

'''
1|Toy Story (1995)|
2|GoldenEye (1995)|
3|Four Rooms (1995)|
4|Get Shorty (1995)|
5|Copycat (1995)|
6|Shanghai Triad (Yao a yao yao dao waipo qiao)
7|Twelve Monkeys (1995)|
'''
smallMatrix = userItemMatrix(smallHandData, nUsers, nItems)
trainSet1 = importTestData('u1.base')
testSet1 = importTestData('u1.test')
allData = importAllRatingData()

allMatrix = userItemMatrix(ratingData , numUsers, numItems)

print itemCF(smallMatrix, 7 ,k = 7)

class Recommender1Test(unittest.TestCase):
    def testImportAllData(self):
        self.assertEquals(allData.shape , (100000, 4))
        self.assertEquals(np.count_nonzero(allData['rating'].unique()) , 5)
        self.assertEquals(np.count_nonzero(allData['user id'].unique()) , 943)
        self.assertEquals(np.count_nonzero(allData['item id'].unique()) , 1682)
     
    def testImportTrainData1(self):
        self.assertEquals(np.count_nonzero(trainSet1['rating'].unique()) , 5)
        self.assertEquals(np.count_nonzero(trainSet1['user id'].unique()) , 943)
        #Train set contains 1650 items 
        self.assertEquals(np.count_nonzero(trainSet1['item id'].unique()) , 1650)
        self.assertEquals(1,1)
        
    def testImportTestData1(self):
        self.assertEquals(np.count_nonzero(testSet1['rating'].unique()) , 5)
        #Test set 1 has 459 users
        self.assertEquals(np.count_nonzero(testSet1['user id'].unique()) , 459)
        #Test set contains 1410 items 
        self.assertEquals(np.count_nonzero(testSet1['item id'].unique()) , 1410)
        self.assertEquals(1,1)    
        
    def testUserItemMatrixRatings(self):
        #check some user ratings to make sure data matrix index is correct 
        self.assertEquals(userItemMatrix(allData,943, 1682)[196][242] , 3) 
        self.assertEquals(userItemMatrix(allData,943, 1682)[253][465] , 5) 
        self.assertEquals(userItemMatrix(allData,943, 1682)[279][64] , 1) 
        #test with hand made set 
        self.assertEqual(smallMatrix[1][1] , 5.0)
        self.assertEqual(smallMatrix[7][7] , 4.0)
        self.assertEqual(smallMatrix[5][7] , 1.0)
      
    def testBaseLine(self):
        self.assertEqual(allData.groupby(['user id'])['rating'].mean()[1],baseline(allMatrix, 1))
        self.assertEqual(allData.groupby(['user id'])['rating'].mean()[495],baseline(allMatrix, 495))
        self.assertEqual(allData.groupby(['user id'])['rating'].mean()[721],baseline(allMatrix, 721))
        #test with hand made set 
        self.assertEqual(baseline(smallMatrix, 4) , 5.0)
        self.assertEqual(baseline(smallMatrix, 1) , 3.6)
        self.assertEqual(baseline(smallMatrix, 7) , 3.0)
        
    def testUserCFRecommender(self):
        #user 1 is very simillar to user 3 and had rated 'Shanghai Triad' with 4 stars
        #A collabrative filter with k = 1 should recommend this movie 
        self.assertEqual(userCFrecommender(smallMatrix, 3 ,k = 1 , n = 1), \
                         [(4.0, 'Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)')])
    def testItemCFRecommender(self):
        self.assertEqual(1,1)
        
if __name__ == '__main__':
    unittest.main()        