'''
Created on Jan 15, 2019

@author: ppc
'''

from classification.custom_nn_base import CustomNNBase
from NN_loss.ordinal_categorical_crossentropy import lossOCCQuadratic, lossOCC

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, InputLayer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing.label import LabelEncoder



class CustomNNCategorical(CustomNNBase):  
    """
    Base for custom sk classifier implementing NN using tf (through keras)
    
    implement an MLP (define in setModel) 
    """
    def __init__(self, hidden = [200,100,50,20], dropout = [0.1,0.1], reg = [], h_act = [tf.nn.relu],
                       epoch = 500, batch_size = 32,
                       cbEarly = EarlyStopping(monitor='acc', min_delta=0.0001, patience=20, verbose=0, mode='auto'),
                       loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy']
                ):
        """
        set shape of the MLP
        :param hidden:
        :param dropout:
        :param reg:
        :param h_act: hidden_actication
        """
        assert loss in ["categorical_crossentropy", lossOCC, lossOCCQuadratic]
        CustomNNBase.__init__(self, epoch, loss, optimizer, metrics, batch_size)
        # 'categorical_crossentropy', OCC.lossOCCQuadratic, lossOCC
        assert (len(hidden) > 0) & (len(hidden) >= len(dropout)) & \
            (len(hidden) >= len(reg)) & (len(hidden) >= len(h_act))
        self.hidden = hidden
        self.dropout = dropout
        self.reg = reg
        self.h_act = h_act
        
        self.cbEarly = cbEarly
        self.final_activation = tf.nn.softmax

    def __compile(self, input_shape, output_shape):
        ter = lambda x,i: None if len(x) <= i else x[i]
        reg = [tf.keras.regularizers.l2(i) for i in self.reg]
        h_act = self.h_act*round(len(self.hidden)/len(self.h_act))
#         RuntimeError: Cannot clone object CustomNNCategorical(batch_size=32,..., reg=[]),
# as the constructor either does not set or modifies parameter h_act
        
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(input_shape,)))
        for i in range(0, len(self.hidden)):
            self.model.add(Dense(self.hidden[i], activation=h_act[i], 
                                kernel_regularizer=ter(reg, i), bias_regularizer=ter(reg, i) ))                       
            if not ter(self.dropout, i) is None: self.model.add(Dropout(ter(self.dropout, i)))
        self.model.add(Dense(output_shape, activation= self.final_activation))
        
        self.model.compile(optimizer = self.optimizer,
              loss = self.loss,
              metrics=self.metrics)

    def __category_to_output(self, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        target = tf.keras.utils.to_categorical(y, num_classes=np.unique(y).size)
        return target
    
    def __output_to_category(self, output):
        pred = [np.argmax(i) for i in output]
        pred = self.label_encoder.inverse_transform(pred)
        return pred

    def fit(self, X, y=None):
        '''
        :param X:
        :param y:
        :param cbEarly: Parameter for early stopping
        '''
        output = self.__category_to_output(y)
        output_shape = output.shape[1]
        input_shape = X.shape[1]
        self.__compile(input_shape, output_shape)
        
        cbReduceLR = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.8, patience=3,
            verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        self.history = self.model.fit(
            X, output,
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [cbReduceLR, self.cbEarly],
            verbose=0)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "history")
        except AttributeError:
            raise RuntimeError("Call fit first.")
        
        preds = self.model.predict(X)
        preds = self.__output_to_category(preds)
        return preds
                       

    def plot_history(self):
        history = self.history
        print("History acc", history.history['acc'])
        print("History loss", history.history['loss'])
        print("History lr", history.history['lr'])
        print("Acc train (last)", history.history['acc'][-5:-1])
        
        plt.plot(history.history['acc'])
        plt.ylabel('history.history["acc"]')
        plt.show()
        plt.plot(history.history['loss'])
        plt.ylabel('history.history["loss"]')
        plt.show()
        plt.plot(history.history['lr'])
        plt.ylabel('history.history["lr"]')
        plt.show()
