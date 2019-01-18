'''
@author: JHart96
https://github.com/JHart96/keras_ordinal_categorical_crossentropy

@note: Does not seem to improve result (ill thought?, bugged? or just equivalent?)

@see tf_mvp.py comment about loss for truly intersting approach
'''

from tensorflow.keras import backend
from tensorflow.keras import losses

def lossOCC(y_true, y_pred):
    weights = backend.cast(backend.abs(backend.argmax(y_true, axis=1) - backend.argmax(y_pred, axis=1))/(backend.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def lossOCCQuadratic(y_true, y_pred):
    '''
    Try to adapt JHart func for quadratic
    :param y_true:
    :param y_pred:
    '''
    weights = backend.cast(backend.square(backend.argmax(y_true, axis=1) - backend.argmax(y_pred, axis=1))/(backend.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)