'''
Created on Jan 20, 2019

@author: ppc

@todo clean code, hardcoded path
'''
import json
import glob, os
import pandas as pd
import numpy as np

import pdb; 
import matplotlib.pyplot as plt


def _load_pet(dir, petId, index, printD=False):
    os.chdir(dir)
    # method to slow while profiling, probably du to very high number of file in dir
    print("issues with glob being slow @todo")
    files =  glob.glob(petId + "*") # each img as its own ggle api run, several img per animal
    if len(files) > index:
    # @warning: filename is not necessarily "-1"
        return files[index] 
    else: 
        return None

def _load_pet_json(dir, petId, index, printD=False):
    filename = _load_pet(dir, petId, index, printD)
    if not filename: return None
    
    with open(dir+filename) as json_file:  
        data = json.load(json_file)
        
    if printD: print(json.dumps(data, indent=4, sort_keys=True))
    return data

#"train_metadata", "test_metadata", "train_sentiment", "test_sentiment"
def parse_metadata(x,dir_met):
    '''
    create dess dataframe with columns "PetID", "label1", "label2" ..., "labelexpected_len"
    the same for score
    :param x:train or test
    :param dir_met:dir for metadata
    :see prep_metadata
    '''
#     print(x.loc[animal,:])
    metadata = {}
    expected_len = 10 # expected len of label annotation in petId json 
    ids = list(x["PetID"])
    count = 0
    for petId in ids:
        print(count)
        print(petId)
        count += 1
        
        data = None
        index = 1
        while (data is None): # search to find desired json and parse it
            try:
                data = _load_pet_json(dir_met, petId, index)
                if data.get("labelAnnotations") is None: data = None # check if json is has expected
                index += 1 # try for all json related to this pet until one match desired format else data = None
                if index > 30: raise "index"
            except Exception as e:
                print(e)
                print('Error for petId ', petId)
                data = None
                break
#                 pdb.set_trace()

        lad = list()
        las = list()
        
        if data:
            try:
                for la in data['labelAnnotations']:
                    lad.append(la['description'])
                    las.append(la['score'])
            except:
                print("patience is a virtue")
                print("petid", petId)
                pdb.set_trace()

        assert len(lad) ==  len(las),\
            "len, las " + str(len(las)) + "lad " + str(len(lad)) + ";json does not follow expected format"
        lad = lad + [""]*(expected_len - len(lad)) # fill lad to be of len 10 (>80% of metadata are 10)
        las = las + [np.nan]*(expected_len - len(las)) # fill las to be of len 10
        metadata[petId] = {'description':lad, "score": las}
    
    dess = [metadata[x]["description"] for x in ids]
    scores = [metadata[x]["score"] for x in ids]
        
    dess =  pd.DataFrame(dess)
    scores = pd.DataFrame(scores)    
    dess.columns = ["label" + str(i) for i in range(0, expected_len)]
    scores.columns = ["label_score" + str(i) for i in range(0, expected_len)]
    dess["PetID"] = ids
    scores["PetID"] = ids
    return dess, scores

def merge_metadata(dir_met, train, test):
    '''
    :see preprocessed/prep_metadata
    :param dir: with train test
    '''
    lenTr = train.shape[0]
    lenTest = test.shape[0]
    dess_train = pd.read_pickle(dir_met + "dess_train" + ".pkl")
    scores_train = pd.read_pickle(dir_met + "scores_train" + ".pkl")
    dess_test = pd.read_pickle(dir_met + "dess_test" + ".pkl")
    scores_test = pd.read_pickle(dir_met + "scores_test" + ".pkl")
    
    train = pd.merge(train, dess_train, how='inner', on=['PetID'])
    train = pd.merge(train, scores_train, how='inner', on=['PetID'])
    test = pd.merge(test, dess_test, how='inner', on=['PetID'])
    test = pd.merge(test, scores_test, how='inner', on=['PetID'])
    
    assert lenTr == train.shape[0], "pkl outdated train"
    assert lenTest == test.shape[0], "pkl outdated test"
    return train, test
    