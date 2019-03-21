'''
Created on Jan 16, 2019

@author: ppc
'''
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin



class CustomNNBase(BaseEstimator, ClassifierMixin):  
    '''
    classdocs
    '''
    def __init__(self, epoch, loss, optimizer, metrics, batch_size):
        '''
        Constructor
        '''
        self.model = None
        
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        
        self.epoch = epoch
        self.batch_size = batch_size
        
    
    def __compile(self, input_shape, output_shape):
        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pass

    def score(self, X, y=None):
#         self.predict(X)
#         res = model.evaluate(x_test, y_test, verbose=0)
        raise "no defined score"
        return None
        