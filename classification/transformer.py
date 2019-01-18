'''
Created on Jan 14, 2019

@author: ppc

Transformer for preprocessing
To use in a pipeline
'''

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, attribute_names, dtype=None):
        self.attribute_names = attribute_names
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_selected = X[self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        return X_selected.values
    
    
    
class PipeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    solve label encoder issues: TypeError: fit_transform() takes 2 positional arguments but 3 were given,
    take more than one column, set unseen label (test) to extra label
    
    @note equivalent: class skutil.preprocessing.SafeLabelEncoder
        
    LabelEncode all column of Xt, ignores y
    """
    def __init__(self, silent = False, astype = None):
        self.values = list()
        super().__init__()
        self.labels = []
        self.silent = silent
        self.astype = astype
        
    def fit(self, Xt, y=None):
        for i in range(0, Xt.shape[1]):
            self.labels.append(np.unique(Xt[:,i]))
        return self
    
    def transform(self, Xt):            
        assert Xt.shape[1] == len(self.labels)
        for i in range(0, Xt.shape[1]):
            # Test set might have values yet unknown to the classifiers
            unknown = np.setdiff1d(np.unique(Xt[:,i]), self.labels[i], assume_unique=True)
            if (len(unknown) > 0) & (not self.silent) : print(len(unknown), "unknown labels found.")
            
            uValues = np.append(self.labels[i], unknown)
            # all unknown values will take the same extra label
            futurLabel = list(range(0, self.labels[i].size)) + [self.labels[i].size]*len(unknown)
            mapping = dict(zip(uValues, futurLabel))
            
            f = lambda i, mapping: mapping[i] 
            Xt[:,i] = np.vectorize(f)(Xt[:,i], mapping)
        if not self.astype is None: Xt = Xt.astype(self.astype)
        return Xt
    
    
    
class inferNA(BaseEstimator, TransformerMixin):
    '''
    infer na to mean of value (even for unordered value because they are all binary)
    '''

    def __init__(self, attribute_names, method="mean"):
        '''
        :param attribute_names: feature for which NAs will be infered
        :param method:
        '''       
        self.method = method 
        self.attribute_names = attribute_names
        self.replacement =  dict()

    def fit(self, X, y=None):
        assert self.method == "mean"
        for i in self.attribute_names:
            self.replacement[i] = X.loc[:,i].mean(skipna=True) 
        return self

    def transform(self, X, y=None):
        X =  X.copy() # pd SettingWithCopyError
        for i in self.attribute_names:
            X.loc[:,i] = X.loc[:,i].replace(np.nan, self.replacement[i], inplace = False)
        return X
