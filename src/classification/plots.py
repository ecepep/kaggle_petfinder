'''
Created on Feb 9, 2019

@author: ppc
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
   
pathToAll = "../all" # path to dataset dir

# train, test = getTrainTest2_meta_img_rn(pathToAll)
train, test = get_from_pkl(pathToAll, v = "getTrainTest2_meta_img_rn")



 
##############################################
# plot pca explaind variance against nb of component for preprocessed img features
##############################################                                    
        
pipe_rdf_img_PCA = pipe_rdf_img_PCA.fit(X = train, y = train["AdoptionSpeed"])        
pca = pipe_rdf_img_PCA.named_steps.u_prep.transformer_list[2][1].named_steps.PCA #or 2
                                                                                         
                                                                                         
plt.figure()                                                                          
plt.plot(np.cumsum(pca.explained_variance_ratio_))                                    
plt.xlabel('Number of Components')                                                    
plt.ylabel('Variance (%)') #for each component                                        
plt.title('pca Explained Variance')                                                   
plt.show()                   
  
# log shape, 50 cpnt explain ~~80% of the variance

##############################################                                                         