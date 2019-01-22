'''
Created on Jan 20, 2019

@author: ppc

parse metadatas' json for train and test petids and save it as a pd.dataframe in .pkl in /preprocessed/ + meta_dir
'''
from classification.util import getTrainTest2
from preprocessed.metadata import parse_metadata
pathToAll = "../all" # path to dataset dir
pathToAll = "/home/ppc/Documents/eclipse-workspace/kaggle_petfinder/all"
meta_dir = "/preprocessed/metadata_label/"

# read train, test csvs, set unknown to NA and shuffle
train, test = getTrainTest2(pathToAll)

    
dess_train, scores_train = parse_metadata(train.iloc[:,:], 
          pathToAll + "/train_metadata/")
dess_test, scores_test = parse_metadata(test.iloc[:,:], 
          pathToAll + "/test_metadata/")


dess_train.to_pickle(pathToAll + meta_dir + "dess_train" + ".pkl")  # where to save it, usually as a .pkl
scores_train.to_pickle(pathToAll + meta_dir + "scores_train" + ".pkl")  # where to save it, usually as a .pkl

dess_test.to_pickle(pathToAll + meta_dir + "dess_test" + ".pkl")  # where to save it, usually as a .pkl
scores_test.to_pickle(pathToAll + meta_dir + "scores_test" + ".pkl")  # where to save it, usually as a .pkl

# dess_train = pd.read_pickle(pathToAll + meta_dir + "dess_train" + ".pkl")
# dess_test = pd.read_pickle(pathToAll + meta_dir + "dess_test" + ".pkl")
# scores_train = pd.read_pickle(pathToAll + meta_dir + "scores_train" + ".pkl")
# scores_test = pd.read_pickle(pathToAll + meta_dir + "scores_test" + ".pkl")


# pathToAll + "/train_sentiment/",
# pathToAll + "/train_images/")
