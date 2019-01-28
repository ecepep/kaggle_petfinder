'''
Created on Jan 22, 2019

@author: ppc

preprocess img to create interesting features for classification by freezing first layer of a pretrained CNN.
'''

from tensorflow.keras.applications import VGG16

be careful with output size not to have to high dim (min 1000?) maybe a pca could help at the end




#Load the VGG model

# time to create a VGG
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
    
    # Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model.layers.pop() # Get rid of the classification layer
model.layers.pop() # Get rid of the dropout layer
model.outputs = [model.layers[-1].output]

from keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
    
class FrozenCnn():
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
    def __compile(self, input_shape, output_shape):
        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pass

    def score(self, X, y=None):
#         self.predict(X)
#         res = model.evaluate(x_test, y_test, verbose=0)
        raise "no defined score"
        return None
        
