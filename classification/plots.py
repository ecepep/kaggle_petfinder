'''
Created on Feb 9, 2019

@author: ppc
'''
from classification.util import getTrainTest2_meta_img_rn, get_from_pkl
from classification.pipelines import pipe_rdf_img_PCA, des_pipe_for_svd

import numpy as np
import matplotlib.pyplot as plt
import os
import cProfile  
   
pathToAll = "../all" # path to dataset dir

# train, test = getTrainTest2_meta_img_rn(pathToAll)
train, test = get_from_pkl(pathToAll, v = "getTrainTest2_meta_img_rn")


# des_pipe_for_svd.fit(X = train, y = train["AdoptionSpeed"])

   
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