'''
Created on Jan 9, 2019

@author: ppc

Probably no need for NN in the context BUT for the sake of learning keras.



https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

@todo
- Bring everything together with sk bench in a clean generic prog + sk_bench todos.
- improve loss for kappa quadratic
- try https://github.com/sdcubber/f-measure/blob/master/src/classifiers/OR.py
- try input [1,1,1,0], binary cross entropy + sigmoid (last layer)
- batch normalization
- increase lr? :/
- choose ini?  :/


keras.initializers.lecun_uniform(seed=None)
keras.initializers.glorot_normal(seed=None) ## Xavier
keras.initializers.he_normal(seed=None) ## He

How can I "freeze" layers?


EarlyStopping
ModelCheckpoint
ReduceLROnPlateau

https://stackoverflow.com/questions/50649831/understanding-regularization-in-keras

Create a callback

keras.utils.to_categorical(y, num_classes=None, dtype='float32')
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, 
show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

improving loss:
- https://github.com/JHart96/keras_ordinal_categorical_crossentropy
- A Neural  Network  Approach  to Ordinal Regression
- Weighted kappa loss function for multi-class classification of ordinal data in deep learning
- A simple squared-error reformulation for ordinal classification
-1 I have one idea for this, which is to calculate loss based on comparing the outputs with 1 0 0 for A, 1 1 0 for B, and 1 1 1 for C.
https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data
-1 https://github.com/sdcubber/f-measure/blob/master/src/classifiers/OR.py

'''

import warnings
import NN_loss.kappa_metrics as km
import NN_loss.ordinal_categorical_crossentropy as OCC

from bench_sk.preprocessing import *
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from tensorflow.python.keras.callbacks import EarlyStopping

#######################################################################

# meta to determined which method to use with either ordered input for y or not
METHOD = "ordered" # "categorical" "ordered"

if (METHOD == "ordered"):
    def toOrderedCategorical(y, num_classes):
        """
        Ordered categorical value  represented as:
        0: [0, 0, 0, 0, 1],1: [0, 0, 0, 1, 1],2: [0, 0, 1, 1, 1]... 
        :param y: array with value between 0 and num_classes - 1
        :param num_classes: 
        """
        assert set(y.unique()) == set(range(0,num_classes)), \
            "rewrite more exhaustive fun (toOrderedCategorical)"     
        ordered = [([0]*(c-i) + [1]*i) for i, c in zip(y, [num_classes]*len(y))]
        return np.array(ordered)
    
    def yToClass(y):
        y = y.round().astype(int)
        return [i.sum() for i in y]

    y_transform = toOrderedCategorical
#     reg = tf.keras.regularizers.l2(0.00005)  # alt to dropout
    reg = None
    finalActivation = tf.nn.sigmoid
    methodLoss = 'binary_crossentropy'
    cbEarly = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0000001,
                                           patience=20, verbose=0, mode='auto')
    
elif(METHOD == "categorical"):
    def yToClass(y):
        return [np.argmax(i) for i in y]

    y_transform = tf.keras.utils.to_categorical
    reg = tf.keras.regularizers.l2(0.00005)
    finalActivation = tf.nn.softmax
    methodLoss = 'categorical_crossentropy'  # OCC.lossOCCQuadratic, lossOCC
    cbEarly = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001,
                                           patience=20, verbose=0, mode='auto')
    
else: raise "METHOD unspecified or unknown"

#######################################################################

train, test = getTrainTest1(pathToAll="../all/", silent = True)

x_train, x_test, y_train, y_test = train_test_split(
        train.drop(["AdoptionSpeed", "PetID"], axis=1),
        train["AdoptionSpeed"], test_size=0.33, random_state=None)

standardizer = Normalizer(norm="l1")
x_train = standardizer.fit_transform(x_train)
x_test = standardizer.fit_transform(x_test)

y_train = y_transform(y_train, num_classes=5)
y_test = y_transform(y_test, num_classes=5)

#####################################################################

model = tf.keras.models.Sequential([
    # default (for awareness):
    # keras.layers.Dense(units, activation=None, use_bias=True, 
    # kernel_initializer='glorot_uniform', bias_initializer='zeros',
    # kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    # kernel_constraint=None, bias_constraint=None)
    tf.keras.layers.Dense(200, activation=tf.nn.relu , input_shape=(x_train.shape[1],)
        , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
    tf.keras.layers.Dropout(0.1), #alt to regulariztion
    tf.keras.layers.Dense(100, activation=tf.nn.relu
        , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(50, activation=tf.nn.relu
    , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation=tf.nn.relu
    , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation= finalActivation)
])

model.compile(optimizer='adam',
              loss = methodLoss,
              metrics=['accuracy'])

cbReduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.8, patience=3,
     verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
# Issue in stopping early for METHOD == "ordered" since accuracy is ~~wrong
cbEarly = cbEarly
#           ,restore_best_weights=True) not in init but in kera's doc? later version of keras?

history = model.fit(x_train, y_train,
                    epochs=500, batch_size=32,
                    callbacks=[cbReduceLR, cbEarly],
                    verbose=0)  # validation_data = (x_test, y_test) 
res = model.evaluate(x_test, y_test, verbose=0)

predsTrain = model.predict(x_train, batch_size=None, verbose=0, steps=None) # check generalization
preds = model.predict(x_test, batch_size=None, verbose=0, steps=None)

#####################################################################

if (METHOD == "ordered"): warnings.warn("Accuracy value will be wrong because using binary crossentropy as loss")

print("History acc", history.history['acc'])
print("History loss", history.history['loss'])
print("History lr", history.history['lr'])
print("Acc train (last)", history.history['acc'][-5:-1])
print('Acc test:', res)

predsTrain = yToClass(predsTrain)
preds = yToClass(preds)
targsTrain = yToClass(y_train)
targs = yToClass(y_test)

kappaScoreTrain = metrics.cohen_kappa_score(targsTrain, predsTrain, weights="quadratic")
print("kappaScore train:", kappaScoreTrain)
kappaScore = metrics.cohen_kappa_score(targs, preds, weights="quadratic")
print("kappaScore:", kappaScore)

plt.plot(history.history['acc'])
plt.ylabel('history.history["acc"]')
plt.show()
plt.plot(history.history['loss'])
plt.ylabel('history.history["loss"]')
plt.show()
plt.plot(history.history['lr'])
plt.ylabel('history.history["lr"]')
plt.show()

"""
conclusion (!= hyperparam opt)

Without dropout overfitting (btw loss take too long to converge) acc train 0.7, test 0.3
with dropout good generalization
dropout few  (0.1,0.1,0,0): train 0.42, test: 0.36
dropout more (0.2,0.2,0.1,0.1): train 0.405, test: 0.38
dropout lot (0.3,0.3,0.2,0.2): train 0.38, test:0.39
still take long to converge:

Try to use regularization in cmp
with reg on bias and kernel 0.00005 similar res  acc train 0.397 acc test 0.374 kappaScore 0.2479551944591608

with diff loss (impact on quadra kappa):
- categorical_crossentropy (reg :0.00005): acc train 0.397 acc test 0.374 kappaScore 0.248 # 0.29 on 2nd try
- categorical_crossentropy (reg :0.00002): acc train 0.454 acc test 0.371 kappaScore 0.269
- OCC.lossOCC (reg :0.00002): acc train 0.391 acc test 0.376 kappaScore 0.269 epoch only 80?
- OCC.lossOCC (reg :0.000002): acc train0.440 acc test 0.357 kappaScore 0.234 
- OCC.lossOCC (reg :0.00005): acc train 0.432 acc test 0.368 kappaScore 0.245 
- OCC.lossOCC (reg :0.0001): acc train 0.379 acc test 0.38 kappaScore 0.271 
(variance in score from rn is higher than reg changes?)
- lossOCCQuadratic (reg :0.00005): poor
- lossOCCQuadratic (reg :0.0005): poor
- lossOCCQuadratic (reg :0.000005): acc train ? acc test ? kappaScore 0.247
=> not helping 

METHOD = "ordered": (without reg)
    kappaScore train: 0.3473411999200494
    kappaScore: 0.2732755044352211
    reg 0.00005 poor 0.24
    (no reg late epoch)
    kappaScore train: 0.46711530844097804
    kappaScore: 0.27509022848628994 
    (no reg, small dropout)
    kappaScore train: 0.3094877306310425
    kappaScore: 0.27617690298776953
METHOD = "categorical(with reg 0.00005)
    kappaScore train: 0.3389869346011669
    kappaScore: 0.28156816168831833
    kappaScore train: 0.3231622843578743
    kappaScore: 0.27562959412181354
"""

