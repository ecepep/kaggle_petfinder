'''
Created on Feb 15, 2019

@author: ppc

:warning draft

@todo 
- a randomized search on everything could be good to replace @file pipeline but need both a sk. child classifier 
with all param and probably will be complicated in compute power because of NN. 
- implement caching
'''
from classification.custom_nn_categorical import CustomNNCategorical

class RS_Classifier(BaseEstimator, ClassifierMixin):  
    '''
    classdocs
    '''
    def __init__(self, clf_type = ["catNN", "orderedNN", "RnForest", "GBT"], paramforcat, paramfororedered....):
        '''
        Constructor
        '''
        if (clf_type == "catNN")
            self.clf = CustomNNCategorical(paramforcat)
    
    def fit(self, X, y=None):
        self.clf.fit(X, y)
        return self
    ...

    
class RS_Preprocessor(BaseEstimator, TransformerMixin): 
    ? should there be some link with RS_classifier to limit scale of possible param? maybe should be included in clf
    