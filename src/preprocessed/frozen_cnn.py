'''
Created on Jan 22, 2019
 
@author: ppc
@todo != image resizing?
@todo code structure relevance? #scikit
 
preprocess img to create interesting features for classification by freezing first layer of a pretrained CNN ("imagenet").
'''
 
from keras.applications import VGG16
from keras.layers import Dropout
from keras import optimizers
from keras.models import Sequential
from keras import backend

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize
import sys
from time import sleep
import numpy as np

class FrozenCnn():
    '''
    classdocs
    Predict img from a frozen pretrained cnn 
    '''
    def __init__(self, cnn = VGG16, inputshape = (224,224), n_pop=1):
        '''
        Constructor
        
        :param cnn: cnn to be loaded with imagenet
        :param inputshape: input shape of the CNN, img will be resized to it 
        :param n_pop: number of popped layer
        '''
        self.cnn = cnn
        self.inputshape = inputshape
        if self.cnn == VGG16 and self.inputshape != (224,224): raise "input shape does not match keras VGG16"
        self.n_pop = n_pop
        
    def __compile(self, input_shape, output_shape):
        pass
 
    def fit(self, X = None, y=None):
        if not X is None: print("frozen does not fit to X! (pretrained on imagenet)")
        
        vgg_conv = self.cnn(weights ='imagenet', include_top=True)
        
        for layer in vgg_conv.layers:
            layer.trainable = False
#             print(layer)
#             print("layer out", layer.output)
        
        model = Sequential()
        model.add(vgg_conv)
        
        for i in range(0, self.n_pop):
            model.layers.pop() # Get rid of the classification layer  
            while type(model.layers[-1]) is Dropout:
                model.layers.pop() # Get rid of the dropout layer
        
        print("output:", model.layers[-1].output)
            
        self.model = model   
        return self
 
    def predict(self, X, y=None):
        '''
        use plt.show to show pictures where resize deform with h or w twice bigger
        :param X:
        :param y:
        '''
        # load pretrained cnn from keras
        
        image_resized = []
        for img in X:
            image_resized.append(imresize(img, self.inputshape))
            if img.shape[0] / img.shape[1] > 2 or img.shape[0] / img.shape[1] < 0.5: 
                print("warning on im resize shape,", img.shape)
#                 fig=plt.figure()
#                 fig.add_subplot(1,2,1)
#                 plt.imshow(img)
#                 fig.add_subplot(2,2,2)
#                 plt.imshow(image_resized[-1])
#                 plt.draw()
#     #             sleep(3)

        prediction = self.model.predict(np.array(image_resized))
#         backend.clear_session() # (was) necessary to avoid memory leakage
        return prediction
           
 
    def score(self, X, y=None):
        pass
    
    def __del__(self):
        backend.clear_session() # could create trouble du to carbage collector being late???
         
