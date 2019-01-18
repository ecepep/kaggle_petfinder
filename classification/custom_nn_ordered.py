'''
Created on Jan 16, 2019

@author: ppc
'''
from classification.custom_nn_categorical import CustomNNCategorical

import tensorflow as tf
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
        print("Accuracy from keras is wrong here")
        self.loss = None
#         assert loss
#         assert cbEarly (if None ==> define else warning acc)
        CustomNNCategorical(*args, **kwargs)

        if not self.loss is None: warnings.warn("loss will be set as 'binary_crossentropy'")
        self.loss = "binary_crossentropy"

        # acc is bad, cb early with lower delta
        if "cbEarly" in kwargs:
            self.cbEarly = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0000001,
                                           patience=20, verbose=0, mode='auto')
        self.final_activation = tf.nn.sigmoid
    
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
        
        