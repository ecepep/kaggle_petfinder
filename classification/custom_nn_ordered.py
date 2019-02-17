'''
Created on Jan 16, 2019

@author: ppc
'''
from classification.custom_nn_categorical import CustomNNCategorical

from tensorflow.nn import sigmoid
import numpy as np
import warnings

class CustomNNordered(CustomNNCategorical):
    '''
    implement an mlp classifier for ordered categorical Ys represented as:
        0: [0, 0, 0, 0, 1],1: [0, 0, 0, 1, 1],2: [0, 0, 1, 1, 1]...
        
    simplistic solution to infer importance of ordering to the NN.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        self.loss = None
#         assert loss
#         assert cbEarly (if None ==> define else warning acc)
        super(CustomNNordered, self).__init__(*args, **kwargs)

        if not self.loss is None: warnings.warn("loss will be set as 'binary_crossentropy'")
        self.loss = "binary_crossentropy"

        # acc is bad, cb early with lower delta
        if "cbEarly" in kwargs:
            warnings.warn("Accuracy from keras is wrong for CustomNNordered. Choose the monitor wisely")
            
        self.final_activation = sigmoid
    
    def __category_to_output(self, y):
        '''
        :param y: array of target ordered categories. Categories must be number from 0 to n-1 (ordered)
        '''
        n_cat = y.unique().size
        assert set(y.unique()) == set(range(0,n_cat)), \
            "rewrite more exhaustive fun (toOrderedCategorical)"     
        target = [([0]*(c-i) + [1]*i) for i, c in zip(y, [n_cat]*len(y))]
        return np.array(target)
    
    def __output_to_category(self, output):
        pred = output.round().astype(int)
        pred =  [i.sum() for i in pred]
        return pred
    
    def plot_history(self, plotname="NN", saving_file=None):
        warnings.warn("Accuracy from keras is wrong for CustomNNordered. Choose the monitor wisely")
        CustomNNCategorical.plot_history(self, plotname, saving_file)
        