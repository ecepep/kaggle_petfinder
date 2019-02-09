'''
Created on Feb 8, 2019

@author: ppc

preprocess img to create interesting features for classification by freezing first layer of a pretrained CNN.

compute in ~6h
'''

from classification.util import getTrainTest2
from preprocessed.metadata import parse_metadata

import pandas as pd
from preprocessed.img_transfer import parse_img
import matplotlib.pyplot as plt

raise 'tedious computation without gpu'
 
pathToAll = "../all" # path to dataset dir
pathToAll = "/home/ppc/Documents/eclipse-workspace/kaggle_petfinder/all"
img_dir = "/preprocessed/transfered_img/"

# du to systematic reshuflling  in getTrainTest2, I always use the same train test by serializing them
# if False:
#     train, test = getTrainTest2(pathToAll) 
#     train.to_pickle(pathToAll + img_dir + "petid_train" + ".pkl") 
#     test.to_pickle(pathToAll + img_dir + "petid_test" + ".pkl") 
 
train = pd.read_pickle(pathToAll + img_dir + "petid_train" + ".pkl")
test = pd.read_pickle(pathToAll + img_dir + "petid_test" + ".pkl")

# part to be reprosseced after bug correction
raise "limited to failed"
train_failed = [38,40,56,57]
test_failed = [1,5,6]

n_pop = 1
train_pkl_path = pathToAll + img_dir + "train"
pred_train = parse_img(x = train.iloc[:,:],
                       dir_img = pathToAll + "/train_images/", 
                       n_pop = n_pop, 
                       pkl_path = train_pkl_path, 
                       chunk_size = 200, 
                       restrict_to_chunk = train_failed)
# pred_train.to_pickle( + ".pkl") 

test_pkl_path = pathToAll + img_dir + "test"
pred_test = parse_img(x = test.iloc[:,:],
                       dir_img = pathToAll + "/test_images/", 
                       n_pop = n_pop, 
                       pkl_path = test_pkl_path, 
                       chunk_size = 200, 
                       restrict_to_chunk = test_failed)


# @todo not yet computed
raise "outdated"
# pred_test.to_pickle(pathToAll + img_dir + "test" + ".pkl") 
# 
# 
# pred_train_3 = parse_img(train.iloc[:2,:], 
#           pathToAll + "/train_images/", n_pop=3)
# pred_train_3.to_pickle(pathToAll + img_dir + "train_pop_3" + ".pkl") 
# pred_test_3 = parse_img(test.iloc[:,:], 
#           pathToAll + "/test_images/", n_pop=3)
# pred_test_3.to_pickle(pathToAll + img_dir + "test_pop_3" + ".pkl") 
# 
# 
 
plt.show()

