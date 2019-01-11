'''
Created on Jan 9, 2019

@author: ppc

Probably no need for NN in the context BUT for the sake of learning keras.



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

References
----------
[1] Agresti, Alan. Categorical data analysis. Vol. 482. John Wiley & Sons, 2003.
[2] http://fa.bianp.net/blog/2013/logistic-ordinal-regression/
[3] http://fa.bianp.net/blog/2013/loss-functions-for-ordinal-regression/
[4] https://github.com/fabianp/minirank/blob/master/minirank/logistic.py (source code)
[5] https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
'''

import NN_loss.kappa_metrics as km
import NN_loss.ordinal_categorical_crossentropy as OCC

from bench_sk.preprocessing import *
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from tensorflow.python.keras.callbacks import EarlyStopping

train, test = getTrainTest1(pathToAll="../all/")

x_train, x_test, y_train, y_test = train_test_split(
        train.drop(["AdoptionSpeed", "PetID"], axis=1),
        train["AdoptionSpeed"], test_size=0.33, random_state=None)

standardizer = Normalizer(norm="l1")
x_train = standardizer.fit_transform(x_train)
x_test = standardizer.fit_transform(x_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

reg = tf.keras.regularizers.l2(0.00005)  # alt to dropout

model = tf.keras.models.Sequential([
    # default (for awareness):
    # keras.layers.Dense(units, activation=None, use_bias=True, 
    # kernel_initializer='glorot_uniform', bias_initializer='zeros',
    # kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    # kernel_constraint=None, bias_constraint=None)
    tf.keras.layers.Dense(200, activation=tf.nn.relu , input_shape=(x_train.shape[1],)
#         , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.3), #alt to regulariztion
    tf.keras.layers.Dense(100, activation=tf.nn.relu
#         , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(50, activation=tf.nn.relu
#       , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation=tf.nn.relu
#       , kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=None
        ),
#     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # OCC.lossOCCQuadratic, lossOCC
              metrics=['accuracy'])

cbReduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.8, patience=3,
     verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
cbEarly = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001,
                                           patience=20, verbose=0, mode='auto')
#           ,restore_best_weights=True) not in init but in kera's doc? later version of keras?

history = model.fit(x_train, y_train,
                    epochs=500, batch_size=32,
                    callbacks=[cbReduceLR, cbEarly],
                    verbose=0)  # validation_data = (x_test, y_test) 
res = model.evaluate(x_test, y_test, verbose=0)

print("History acc", history.history['acc'])
print("History loss", history.history['loss'])
print("History lr", history.history['lr'])

print("Acc train (last)", history.history['acc'][-5:-1])
print('Acc test:', res)

pred = model.predict(x_test, batch_size=None, verbose=0, steps=None)
pred = [np.argmax(res) for res in pred]
targ = [np.argmax(real) for real in y_test]
kappaScore = metrics.cohen_kappa_score(targ, pred, weights="quadratic")
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

"""

