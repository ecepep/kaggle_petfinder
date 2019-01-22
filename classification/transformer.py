'''
Created on Jan 14, 2019

@author: ppc

Transformer for preprocessed
To use in a pipeline
'''

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse


class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, attribute_names, dtype=None, ravel = True):
        '''
        
        :param attribute_names:
        :param dtype: output type
        :param ravel: convert output shape of (n, 1) to (n,)
        '''
        self.attribute_names = attribute_names
        self.dtype = dtype
        self.ravel = ravel

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_selected = X[self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        if self.ravel & (X_selected.shape[1] == 1):
            return X_selected.values.ravel()
        return X_selected.values
    
class StringConcat(BaseEstimator, TransformerMixin):
    '''
    concat several string features to a single string 
    '''
    def __init__(self, sep = " "):
        '''
        :param sep: separator
        '''
        self.sep = sep

    def fit(self, X, y=None):
        return self
    
    def _concat(self, s):
        return self.sep.join(s)
    
    def transform(self, X):
        return X.apply(self._concat, axis = 1)
        

class FnanToStr(BaseEstimator, TransformerMixin):
    '''
    replace float nan to ""
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def on_array(x_str):
        '''
        replace float nan to ""
        :param x: 1D array with bool sel
        '''
        # print(list(a[v_is_str(a)])[1] == float("nan")) # false, why??
        is_str = lambda x: not type(x) is str
        v_is_str = np.vectorize(is_str)
        x_str[v_is_str(x_str)] = ""
        return x_str

    def transform(self, X):
#         train.Description.fillna("none")
        if len(X.shape) == 1:        
            return FnanToStr.on_array(X)
        else:
            for i in range(0, X.shape[1]):
                X[:, i] = FnanToStr.on_array(X[:, i])
            return X
        
class Formater(BaseEstimator, TransformerMixin):
    '''
    Base for transformer to change format of X after transform in Pipe
    '''
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X 
    
class ToSparse(Formater):
    def transform(self, X):
        return sparse.csr_matrix(X) 
    
class AsType(Formater):
    def __init__(self, astype):
        self.astype = astype
    def transform(self, X):
        return X.astype(self.astype)

class PipeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    solve label encoder issues: TypeError: fit_transform() takes 2 positional arguments but 3 were given,
    take more than one column, set unseen label (test) to extra label
    
    @note equivalent: class skutil.preprocessed.SafeLabelEncoder
        
    LabelEncode all column of Xt, ignores y
    """
    def __init__(self, silent = False):
        self.values = list()
        super().__init__()
        self.labels = []
        self.silent = silent
        
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
        return Xt
    
class PipeOneHotEncoder(PipeLabelEncoder):
    """
    Extend PipeLabelEncoder to one hot
    :warning not using sparse matrix, but np 2D
    OneHotEncode all column of Xt, ignores y
    """
    def __init__(self, silent = False):
        PipeLabelEncoder.__init__(self, silent = False)
    
    def fit(self, Xt, y=None):
        PipeLabelEncoder.fit(self, Xt, y)
        self.nums_label = [len(self.labels[i]) for i in range(0,len(self.labels))] # number of label
        return self
        
    def transform(self, Xt):            
        assert Xt.shape[1] == len(self.labels)
        Xt = PipeLabelEncoder.transform(self, Xt)
        # could use sparse matrix directly
        XtOH = np.zeros((Xt.shape[0], sum(self.nums_label)+1)) # Xt in one hot encode notation
        cumsum_len = np.cumsum([0] + self.nums_label[:-1])
        for ji in range(0, Xt.shape[1]):
            for i in range(0, Xt.shape[0]):
                if not Xt[i,ji] > self.nums_label[ji]: # 'unknown' label is still coded as [0,0,0,0,0,0] 
                    XtOH[i, cumsum_len[ji]+Xt[i,ji]] = 1
        
        return XtOH
    
    
class InferNA(BaseEstimator, TransformerMixin):
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
