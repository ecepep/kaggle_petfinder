'''
Created on Jan 20, 2019

@author: ppc

'''
import json
import glob, os
import pandas as pd
import numpy as np

import pdb; 
import matplotlib.pyplot as plt


def _load_pet(dir, petId, index, printD=False):
    os.chdir(dir)
    files =  glob.glob(petId + "*") # each img as its own ggle api run, several img per animal
    if len(files) > index:
    # @warning: filename is not necessarily "-1"
        filename = files[index] 
    else: 
        return None
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
                data = _load_pet(dir_met, petId, index)
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
#                 petId
# 'dd4b67059'

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
    
    train = pd.merge(train, dess_train, how='inner', on=['PetID', 'PetID'])
    train = pd.merge(train, scores_train, how='inner', on=['PetID', 'PetID'])
    test = pd.merge(test, dess_test, how='inner', on=['PetID', 'PetID'])
    test = pd.merge(test, scores_test, how='inner', on=['PetID', 'PetID'])
    
    assert lenTr == train.shape[0], "pkl outdated train"
    assert lenTest == test.shape[0], "pkl outdated test"
    return train, test
    
    
# 14098
# >>> petId
# '458e8f61c'
# {
#     "cropHintsAnnotation": {
#         "cropHints": [
#             {
#                 "boundingPoly": {
#                     "vertices": [
#                         {},
#                         {
#                             "x": 202
#                         },
#                         {
#                             "x": 202,
#                             "y": 359
#                         },
#                         {
#                             "y": 359
#                         }
#                     ]
#                 },
#                 "confidence": 0.79999995,
#                 "importanceFraction": 1
#             }
#         ]
#     },
#     "imagePropertiesAnnotation": {
#         "dominantColors": {
#             "colors": [
#                 {
#                     "color": {
#                         "blue": 203,
#                         "green": 245,
#                         "red": 251
#                     },
#                     "pixelFraction": 0.048006807,
#                     "score": 0.16060464
#                 },
#                 {
#                     "color": {
#                         "blue": 13,
#                         "green": 15,
#                         "red": 18
#                     },
#                     "pixelFraction": 0.30262518,
#                     "score": 0.05188811
#                 },
#                 {
#                     "color": {
#                         "blue": 30,
#                         "green": 77,
#                         "red": 121
#                     },
#                     "pixelFraction": 0.0070491005,
#                     "score": 0.049773432
#                 },
#                 {
#                     "color": {
#                         "blue": 90,
#                         "green": 119,
#                         "red": 137
#                     },
#                     "pixelFraction": 0.011910549,
#                     "score": 0.038003173
#                 },
#                 {
#                     "color": {
#                         "blue": 21,
#                         "green": 57,
#                         "red": 34
#                     },
#                     "pixelFraction": 0.004861449,
#                     "score": 0.013653678
#                 },
#                 {
#                     "color": {
#                         "blue": 180,
#                         "green": 241,
#                         "red": 252
#                     },
#                     "pixelFraction": 0.028804084,
#                     "score": 0.14307095
#                 },
#                 {
#                     "color": {
#                         "blue": 184,
#                         "green": 250,
#                         "red": 250
#                     },
#                     "pixelFraction": 0.010695187,
#                     "score": 0.100757524
#                 },
#                 {
#                     "color": {
#                         "blue": 56,
#                         "green": 103,
#                         "red": 146
#                     },
#                     "pixelFraction": 0.007778318,
#                     "score": 0.034790304
#                 },
#                 {
#                     "color": {
#                         "blue": 120,
#                         "green": 150,
#                         "red": 168
#                     },
#                     "pixelFraction": 0.01093826,
#                     "score": 0.034108765
#                 },
#                 {
#                     "color": {
#                         "blue": 46,
#                         "green": 47,
#                         "red": 49
#                     },
#                     "pixelFraction": 0.15240642,
#                     "score": 0.03251812
#                 }
#             ]
#         }
#     }
# }
# {'imagePropertiesAnnotation': {'dominantColors': {'colors': [{'color': {'red': 251, 'green': 245, 'blue': 203}, 'score': 0.16060464, 'pixelFraction': 0.048006807}, {'color': {'red': 18, 'green': 15, 'blue': 13}, 'score': 0.05188811, 'pixelFraction': 0.30262518}, {'color': {'red': 121, 'green': 77, 'blue': 30}, 'score': 0.049773432, 'pixelFraction': 0.0070491005}, {'color': {'red': 137, 'green': 119, 'blue': 90}, 'score': 0.038003173, 'pixelFraction': 0.011910549}, {'color': {'red': 34, 'green': 57, 'blue': 21}, 'score': 0.013653678, 'pixelFraction': 0.004861449}, {'color': {'red': 252, 'green': 241, 'blue': 180}, 'score': 0.14307095, 'pixelFraction': 0.028804084}, {'color': {'red': 250, 'green': 250, 'blue': 184}, 'score': 0.100757524, 'pixelFraction': 0.010695187}, {'color': {'red': 146, 'green': 103, 'blue': 56}, 'score': 0.034790304, 'pixelFraction': 0.007778318}, {'color': {'red': 168, 'green': 150, 'blue': 120}, 'score': 0.034108765, 'pixelFraction': 0.01093826}, {'color': {'red': 49, 'green': 47, 'blue': 46}, 'score': 0.03251812, 'pixelFraction': 0.15240642}]}}, 'cropHintsAnnotation': {'cropHints': [{'boundingPoly': {'vertices': [{}, {'x': 202}, {'x': 202, 'y': 359}, {'y': 359}]}, 'confidence': 0.79999995, 'importanceFraction': 1}]}}

    