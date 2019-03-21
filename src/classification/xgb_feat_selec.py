'''
Created on Mar 17, 2019

@author: ppc

# quick test for xgboost features selection


# https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
# try to select features based on the xgb fit. Train then a model (either xgb or rf) from subset of features (with !=
threshold).

Results show that it does not seem to work. The less features,  the smaller score
'''

from classification.util import getTrainTest2_meta_img_rn, get_from_pkl,\
    quadratic_cohen_kappa

import numpy as np
import matplotlib.pyplot as plt
import os
import cProfile  
from classification.transformer import DimPrinter, InferNA, \
    DataFrameSelector, ColorBreedOH, PipeOneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection._split import train_test_split
from classification.pipelines_rdf import pipe_rdf_des, to_xgb
from tensorflow.python.ops.gen_array_ops import deep_copy
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection.from_model import SelectFromModel
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from copy import deepcopy
   
pathToAll = "../all" # path to dataset dir

# train, test = getTrainTest2_meta_img_rn(pathToAll)
train, test = get_from_pkl(pathToAll, v = "getTrainTest2_meta_img_rn")

x_train, x_test, y_train, y_test = train_test_split(train, train["AdoptionSpeed"], test_size = 0.3) 

pipe = pipe_rdf_des      
pipe = to_xgb(pipe)
xgb_clf = to_xgb(Pipeline([('clf', KNeighborsClassifier(2))])) # return a pipeline with the same xgb as in pipe and without prep

pipe.fit(x_train, y_train)    
print("pipe.named_steps.clf.feature_importances_",pipe.named_steps.clf.feature_importances_)

prep_pipe = deepcopy(pipe)
prep_pipe.steps = prep_pipe.steps[:-1]
x_train_tf = prep_pipe.transform(x_train)
x_test_tf = prep_pipe.transform(x_test)    

# X_train_transf = pipe.transform(x_train)
# X_test_transf = pipe.transform(x_test)
THRESHOLD = [0, 0.0005, 0.001,0.002, 0.003]
for th in THRESHOLD:
    
    
    selection = SelectFromModel(pipe.named_steps.clf, threshold=th, prefit=True)
    select_X_train = selection.transform(x_train_tf)
    select_X_test = selection.transform(x_test_tf)
    # train model
    selection_model = xgb_clf
    selection_model2 = RandomForestClassifier(n_estimators = 200)
    selection_model.fit(select_X_train, y_train)
    selection_model2.fit(select_X_train, y_train)
    # eval model
    
    select_pred_train = selection_model.predict(select_X_train)                                                
    select_pred_test = selection_model.predict(select_X_test)      
    select_pred_train2 = selection_model2.predict(select_X_train)                                                
    select_pred_test2 = selection_model2.predict(select_X_test)                                                  
                                                                         
    select_score_train = quadratic_cohen_kappa(select_pred_train, y_train)                                         
    select_score_test = quadratic_cohen_kappa(select_pred_test, y_test)  
    select_score_train2 = quadratic_cohen_kappa(select_pred_train2, y_train)                                         
    select_score_test2 = quadratic_cohen_kappa(select_pred_test2, y_test)  
    
    print("for threshold", th)
    print("select_score_train", select_score_train)
    print("select_score_test", select_score_test) 
    print("select_score_train rdf", select_score_train2)
    print("select_score_test rdf", select_score_test2) 
    
    
    
    

