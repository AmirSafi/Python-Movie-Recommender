# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:36:53 2017

@author: Amir
"""

class Movie:
    def __init__(self , movieID , movieTitle, releaseDate , IMDbURL):
        self.movieID = movieID
        self.movieTitle = movieTitle
        self.releaseDate = releaseDate 
        self.IMDbURL = IMDbURL
        
    def getMovieID(self):
        return self.movieID    
    
    def setMovieTitle(self, title):
        self.movieTitle = title
        
    def toString(self):
        return str(self.movieID) + ' ' + self.movieTitle + ' ' + str(self.releaseDate)