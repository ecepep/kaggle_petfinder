'''
Created on Jan 15, 2019

@author: ppc


PROBLEM WITH averaging of kappas' score in keras make result wrong for early stopping
https://github.com/keras-team/keras/issues/8607


'''

from classification.custom_nn_base import CustomNNBase
from NN_loss.ordinal_categorical_crossentropy import lossOCCQuadratic, lossOCC

from sklearn.preprocessing.label import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, InputLayer
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.nn import relu, softmax
import tensorflow as tf

import numpy as np
from time import sleep
from keras import regularizers
from sklearn import metrics as metrics
from sklearn.model_selection._split import train_test_split
from classification.Cohen_kappa_logger import Cohen_kappa_logger


class CustomNNCategorical(CustomNNBase):  
    """
    Base for custom sk classifier implementing NN using keras
    
    implement an MLP for classification with custom metric cohen kappa, no custom loss for now @todo
    """

    def __init__(self, hidden=[200, 100, 50, 20], dropout=[0.1, 0.1], reg=[0.05,0.05], h_act=[relu],
                       epoch=500, batch_size=32,  cbEarly="metric", loss="categorical_crossentropy", 
                       optimizer='adam', metrics=['cohen_kappa'], kappa_weights="quadratic",
                       validation = 0.2, smooth_cb = True
                ):
        '''        
        :param hidden:
        :param dropout:  dropout[0] is assigned to input then hidden
        :param reg: ularization 
        :param h_act: hidden_actication
        :param epoch:
        :param batch_size:
        :param cbEarly: "metric" or an EarlyStopping instance
        :param loss:
        :param optimizer:
        :param metrics: "Accuracy" or 'cohen_kappa'
        :param kappa_weights: compatible with sk(ex:"quadratic", None) ignored if metrics != 'cohen_kappa'
        :param smooth_cb: if True EarlyStopping use val_cohen_kappa smoothed (left avg window 3),
         only with val_cohen_kappa
        
        :note restore_best_weights requires keras 2.2.3
        
        '''
        assert loss in ["categorical_crossentropy", lossOCC, lossOCCQuadratic]
        CustomNNBase.__init__(self, epoch, loss, optimizer, metrics, batch_size)
        # 'categorical_crossentropy', OCC.lossOCCQuadratic, lossOCC
        assert (len(hidden) > 0) & (len(hidden)+1 >= len(dropout)) & \
            (len(hidden) >= len(reg)) & (len(hidden) >= len(h_act))
            
        self.hidden = hidden
        self.dropout = dropout
        self.reg = reg
        self.h_act = h_act
        self.validation = validation
        
        self.final_activation = softmax
        
        self.cbEarly = cbEarly
        self.smooth_cb = smooth_cb
        
        
        raise add a reduce lr on loss increase
        self.cbReduceLR = ReduceLROnPlateau(
            monitor='loss', factor=0.8, patience=3,
            verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        self.kappa_weights = kappa_weights
        if len(self.metrics) > 1 : raise "nope"
    
        
    def __compile(self, input_shape, output_shape):
        ter = lambda x, i: None if len(x) <= i else x[i]
        reg = [regularizers.l2(i) for i in self.reg]
        h_act = self.h_act * round(len(self.hidden) / len(self.h_act))
#         RuntimeError: Cannot clone object CustomNNCategorical(batch_size=32,..., reg=[]),
# as the constructor either does not set or modifies parameter h_act
        
        self.model = Sequential()
        
        self.model.add(InputLayer(input_shape=(input_shape,)))
        if not ter(self.dropout, 0) is None: self.model.add(Dropout(ter(self.dropout, 0)))
        
        for i in range(0, len(self.hidden)):
            self.model.add(Dense(self.hidden[i], activation=h_act[i],
                                kernel_regularizer=ter(reg, i), bias_regularizer=ter(reg, i)))                       
            if not ter(self.dropout, i+1) is None: self.model.add(Dropout(ter(self.dropout, i+1))) # first for input
        
        self.model.add(Dense(output_shape, activation=self.final_activation))
        
        self.model.compile(optimizer=self.optimizer,
              loss=self.loss,
              metrics=self.metrics)

    def __category_to_output(self, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        target = to_categorical(y, num_classes=np.unique(y).size)
        return target
    
    def __output_to_category(self, output):
        pred = [np.argmax(i) for i in output]
        pred = self.label_encoder.inverse_transform(pred)
        return pred

    def cohen_kappa_metric_keras(self, y_true, y_pred):
        '''
        Do not work as a metric because kappa is not linear and keras make a weighted avg of batches score
        :deprecated @see Cohen_kappa_logger
        '''
        raise "deprecated @see Cohen_kappa_logger"
        return tf.py_func(self.cohen_kappa_score, [y_true, y_pred], tf.float32)       
    
    def cohen_kappa_score(self, y_true, y_pred): 
        raise "deprecated @see Cohen_kappa_logger"
        y_pred = self.__output_to_category(y_pred)
        y_true = self.__output_to_category(y_true)
        
        score = metrics.cohen_kappa_score(y_true, y_pred, weights=self.kappa_weights)
        return score.astype(np.float32)

    def break_on_epoch_n(self, threshold, sec = 60):
        self.n_epoch = len(self.history.history["loss"])
        if self.n_epoch > threshold:
            sleep(sec)  # cool down
    
    def _fit_val(self, X, output):
        # @todo clean below
        if type(self.validation) is float:
            self.history = self.model.fit(
                X, output,
                validation_split = self.validation,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
            
        elif type(self.validation) is tuple:
            assert self.validation[0].shape[1] == X.shape[1], "X_validation must be transformed with prep first"
            self.validation = (self.validation[0], self.__category_to_output(self.validation[1]))
            self.history = self.model.fit(
                X, output,
                validation_data = self.validation,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
            
        elif self.validation is None:
            self.history = self.model.fit(
                X, output,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
        else: raise "unknown validation type"
    
    def _kappa_disambiguation(self, X, output):
        '''
        :param X:
        :param output:
        '''
        self.metric_plot = None
        self.patience = 20 #for cbEarly is enoughfrom observation
                    
        if self.metrics[0] == "accuracy":
            self.metric_plot = "acc"
            raise "min_delta must be redefined according to val_acc"
            if self.use_smooth_cb:
                raise 'not available for acc self.use_smooth_cb'
            if self.cbEarly == "metric":
                self.cbEarly = EarlyStopping(
                    monitor= 'val_acc' if self.validation else "acc", min_delta=0.0001, 
                    patience=self.patience, verbose=0, mode='auto')
            self.kappa_logger = None
            
        elif self.metrics[0] == 'cohen_kappa':
            self.metrics = None # 'cohen_kappa_metric' cannot be supported @see explication in Cohen_kappa_logger
            self.metric_plot = 'cohen_kappa'
            if self.cbEarly == "metric":
                if self.validation:
                    monitor = "val_cohen_kappa_smoothed" if self.smooth_cb else "val_cohen_kappa" 
                else:
                    if not self.smooth_cb:
                        monitor = "cohen_kappa"
                    else: raise "No cohen_kappa_smoothed"
                self.cbEarly = EarlyStopping(
                                    monitor='val_cohen_kappa' if self.validation else "cohen_kappa",
                                    min_delta=0.00000001, patience=self.patience, # a large patience is necessary!
                                    verbose=0, mode='max', restore_best_weights=True)
            
            if type(self.validation) is float:
                X, X_val, output, y_val = train_test_split(X, output, test_size = self.validation)
            elif type(self.validation) is tuple:
                assert self.validation[0].shape[1] == X.shape[1], "X_validation must be transformed with prep first"
                X_val = self.validation[0]
                y_val = self.__category_to_output(self.validation[1])
            elif not self.validation is None: raise "unknown validation type"
            
            self.validation = None

            self.kappa_logger = Cohen_kappa_logger(
                 output_to_category=self.__output_to_category,
                 X_train = X, y_train = output, 
                 X_val = X_val, y_val = y_val, 
                 kappa_weights = self.kappa_weights)
            
        else: 
            print(self.metrics[0])
            raise "not implemented"
        return X, output
            
    def fit(self, X, y=None):
        '''
        :param X:
        :param y:
        :param cbEarly: Parameter for early stopping
        '''
        output = self.__category_to_output(y)
        
        X, output = self._kappa_disambiguation(X, output)
        
        output_shape = output.shape[1]
        input_shape = X.shape[1]
        self.__compile(input_shape, output_shape)
      
        self.callback_list = []
        for cb in [self.kappa_logger, self.cbReduceLR, self.cbEarly]:
            if cb: self.callback_list.append(cb)
        
        self._fit_val(X, output)
        
        self.break_on_epoch_n(50)          
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "history")
        except AttributeError:
            raise RuntimeError("Call fit first.")
        
        preds = self.model.predict(X)
        preds = self.__output_to_category(preds)
        return preds

    def plot_history(self, plotname="NN", saving_file=None):
        '''
        :param plotname:
        :param saving_file: filename where to save plots
        :return plt , to avoid carbage collection and closing of the windows
        '''
        history = self.history
#         print("History acc", history.history['acc'])
#         print("History loss", history.history['loss'])
#         print("History lr", history.history['lr'])
#         print("Acc train (last)", history.history['acc'][-5:-1])

        import matplotlib.pyplot as plt
        
        plt.ion()
        plt.show()

        fig = plt.figure()
        plt.grid(True)
        plt.title(plotname)
#         print("possible plot", history.history.keys())
        if self.metric_plot in history.history.keys():
            plt.subplot(221)
            plt.plot(history.history[self.metric_plot])
            plt.ylabel(self.metric_plot + "  ")
            plt.draw()
        
        if "val_" + self.metric_plot in history.history.keys():
            plt.subplot(222)
    #         print("possible plot", history.history.keys())
            plt.plot(history.history["val_" + self.metric_plot])
            plt.ylabel("val_" + self.metric_plot + "  ")
            plt.draw()
    
            if True: 
                print("self.patience last epochs")
                print(history.history["val_" + self.metric_plot][-(self.patience+1):])            
        
        plt.subplot(223)
        plt.plot(history.history['loss'])
        plt.ylabel('"loss" ' + "  " + plotname)
        plt.draw()
        
        plt.subplot(224)
        if "val_cohen_kappa_smoothed" in history.history.keys():
            plt.plot(history.history['val_cohen_kappa_smoothed'])
            plt.ylabel("val_cohen_kappa_smoothed")
        else:
            plt.plot(history.history['lr'])
            plt.ylabel('"lr"' + "  " + plotname)
        plt.draw()
        plt.pause(1)

        if saving_file:
            fig.savefig(saving_file)
       
        return plt
       
    
