'''
Created on Jan 22, 2019

@author: ppc

@todo _None_to_black is ill thought should be set to 0 after CNN

@todo clean code, hardcoded path

'''

import cProfile

from preprocessed.frozen_cnn import FrozenCnn
from preprocessed.metadata import _load_pet
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import glob, os
from time import sleep, time


def _load_pet_img(dir, petId, printD=False):
#     pr = cProfile.Profile()
#     pr.enable()
    #could use _load_pet but to slow
    filename = dir + "/" + petId + "-1.jpg" # always download first pic
    if not os.path.isfile(filename): filename = None 
    if filename is None: return None
    img = mpimg.imread(filename)
#     pr.disable()
#     pr.print_stats()
    return img

def _wrong_shape_to_None(img, shape):
    '''
    to avoid format issues wrong shape are send to None (and afterward black),
    can occur for instance if input img is black and white (we could use another index than 1 too)
    :param img:
    :param shape:
    '''
    if img is None or len(shape) != len(img.shape): return None
    for i in range(0, len(shape)):
        if shape[i] and shape[i] != img.shape[i]: return None
    return img

def _None_to_black(img, shape):
    for i in range(0, len(img)):
        if img[i] is None: img[i] = np.zeros(shape)
    return img 

def parse_img(x, dir_img, n_pop = 1, pkl_path = None, 
              chunk_size = 150, restrict_to_chunk = None):
    '''
    precompute img with frozen cnn in VGG16 
    save chunk of img as pd.df to pkl
    :param x:
    :param dir_img:
    :param n_pop:
    :param pkl_path: path where to save intermediate pkl
    '''
    petIds = np.array((x["PetID"]))
    fcnn = FrozenCnn(cnn = VGG16, inputshape = (224,224), n_pop=n_pop)
    fcnn.fit()
    
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    prediction = list()    
    chunk_n = 0
    for ids in chunks(petIds, chunk_size):
        
        print("count", chunk_n * chunk_size)
        chunk_n += 1
        if restrict_to_chunk and not chunk_n-1 in restrict_to_chunk:
            continue

        try:
            lpi_v = np.vectorize(_load_pet_img, otypes = [object])
            img = lpi_v(dir = dir_img, petId = ids)
        #     if any(np.vectorize(lambda x: x is None)(img)): raise BaseException("some img are None")
            wstn_v = np.vectorize(_wrong_shape_to_None, excluded=['shape'], otypes = [object])
            img = wstn_v(img = img, shape = (None,None,3))
            img = _None_to_black(img, (224,224,3))
                    
            prediction = list(fcnn.predict(img))
                    
            prediction = pd.DataFrame(prediction)
            prediction["PetID"] = ids
        
            pkl_to = pkl_path + "_pop_"  + str(n_pop) + "_n" + str(chunk_n-1) + ".pkl"
            prediction.to_pickle(pkl_to)
        except BaseException as e:
            with open(pkl_path + "_exceptions.txt", "a") as exc_file:
                exc_file.write("="*50 + "\n")
                exc_file.write(str(e)+ "\n")
                exc_file.write("_"*10+ "\n")
                exc_file.write("chunk " + str(chunk_n-1) + " failed"+ "\n")
                exc_file.write("_"*10+ "\n")
                exc_file.write(str(ids)+ "\n")
                   
        sleep(round(chunk_size/2))
    return None
  
def _concat_img(img_dir, set_name, n_pop):
    '''
    concat all pkl's of a set train or test for a certain n_pop
    :param dir_img:
    :param set:
    :param n_pop:
    '''
    from_pkl = img_dir + set_name + "_pop_"  + str(n_pop) + "_n[0-9]*.pkl"
    files =  glob.glob(from_pkl)
    imgs = np.vectorize(pd.read_pickle)(files)
    imgs = pd.concat(imgs, ignore_index=True, sort=False)
    imgs.columns = ["imgf_" + str(i) for i in imgs.columns]
    return imgs
    

def merge_img_fcnn(img_dir, train, test, n_pop = 1):
    '''
    :see preprocessed/prep_metadata
    :param dir: with train test
    '''
    lenTr = train.shape[0]
    lenTest = test.shape[0]
    
    # concat all pkl's
    img_train = _concat_img(img_dir, set_name = "train", n_pop = n_pop)
    img_test = _concat_img(img_dir, set_name = "test", n_pop = n_pop)
    
    train = pd.merge(train, img_train, how='inner', left_on='PetID', right_on='imgf_PetID')
    test = pd.merge(test, img_test, how='inner', left_on='PetID', right_on='imgf_PetID')
    
    assert lenTr == train.shape[0], "pkl outdated train"
    assert lenTest == test.shape[0], "pkl outdated test"
    return train, test
          
    