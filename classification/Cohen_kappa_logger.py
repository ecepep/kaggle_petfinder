'''
Created on Feb 15, 2019

@author: ppc

https://github.com/keras-team/keras/issues/8607

Implemented to replace Custom_NN_categorical.cohen_kappa_metric_keras
because kappa is not linear and keras make
a weighted avg of batches score with custom metrics...
'''

import numpy as np
from sklearn.metrics.classification import cohen_kappa_score as sk_cohen_kappa_score
from keras.callbacks import Callback
from numpy import mean

class Cohen_kappa_logger(Callback):
    '''
    Add to the logs "val_cohen_kappa" and "cohen_kappa" at each epoch's end to record cohen's kappa score.
    Works fine along with EarlyStopping
    '''
    def __init__(self, output_to_category=None,
                 X_train = None, y_train = None, 
                 X_val = None, y_val = None, 
                 kappa_weights = "quadratic",
                 smooth_window = 5):
        '''
        
        :param output_to_category:
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :param kappa_weights:
        :param smooth_window: help avoiding overfitting with earlier early stopping
        '''
        self.output_to_category = output_to_category or (lambda x: x)
        self.X_val = X_val
        self.X_train = X_train
        self.y_val = y_val
        self.y_train = y_train
        self.kappa_weights = kappa_weights
        self.smooth_window = smooth_window
        if X_val is None: raise 'implement none X_val COhen kappa logger'
        

    def on_epoch_end(self, epoch, logs={}):
        pred_train = self.model.predict(self.X_train)
        pred_val = self.model.predict(self.X_val)
        
        score_train = self.cohen_kappa(pred_train, self.y_train)
        score_val = self.cohen_kappa(pred_val, self.y_val)
        
        # or \
#         len(self.model.history.history["val_cohen_kappa"]) == 0
        if not "val_cohen_kappa" in self.model.history.history.keys(): # not defined on first run
            smoothed_val_score = score_val # mean(self.model.history.history["val_cohen_kappa"][-(self.smooth_window+1):])
        else: 
            last_score = self.model.history.history["val_cohen_kappa"][-(self.smooth_window-1):]
            last_score.append(score_val)
            smoothed_val_score = mean(last_score)
            
        logs["cohen_kappa"]= np.float64(score_train)
        logs["val_cohen_kappa"]= np.float64(score_val)
        logs["val_cohen_kappa_smoothed"]= np.float64(smoothed_val_score)
        return

    def cohen_kappa(self, y_true, y_pred): 
        y_pred = self.output_to_category(y_pred)
        y_true = self.output_to_category(y_true)
           
        return sk_cohen_kappa_score(y_true, y_pred, weights=self.kappa_weights)

    


