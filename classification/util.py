'''
Created on Jan 14, 2019

@author: ppc
'''

from pprint import pprint
from time import time, sleep
from sklearn.model_selection._split import  train_test_split
import pdb
import sys
from traceback import print_tb
from sklearn.model_selection._search import GridSearchCV
from random import sample
import sklearn.metrics as metrics

from preprocessed.img_transfer import merge_img_fcnn
from preprocessed.metadata import merge_metadata
from bench_sk.preprocessing import readBaseCSV
from _pickle import dump, load
from os.path import isfile
import matplotlib.pyplot as plt
from copy import deepcopy
from classification.transformer import DimPrinter
from pdb import set_trace
import numpy as np
from sklearn.utils.deprecation import DeprecationDict


def getTrainTest2(pathToAll = "../", silent = True):
    trainPath = pathToAll + "/train.csv"
    if (not silent): print("trainPath ", trainPath)
    testPath = pathToAll + "/test/test.csv"
    if (not silent): print("testPath ", testPath)

    # read csv and set some to na
    train = readBaseCSV(trainPath, silent = silent, shuffle = True, dropText = False,
                  isNa = {"Health" : 0, "MaturitySize" : 0, "FurLength" : 0, "Gender" : 3,
                                "Vaccinated" : 3, "Dewormed" : 3, "Sterilized" : 3})
    test = readBaseCSV(testPath, silent = silent, shuffle = True, dropText = False,
                  isNa = {"Health" : 0, "MaturitySize" : 0, "FurLength" : 0, "Gender" : 3,
                                "Vaccinated" : 3, "Dewormed" : 3, "Sterilized" : 3})
    return train, test

def getTrainTest2_meta_img_rn(pathToAll = "../"):
    meta_dir = pathToAll + "/preprocessed/metadata_label/"
    img_dir = pathToAll + "/preprocessed/transfered_img/"
    
    # read train, test csvs, set unknown to NA and shuffle
    train, test = getTrainTest2(pathToAll)
    # add features preprocessed from metadata dir :see preprocessed/prep_metadata
    train, test = merge_metadata(meta_dir, train, test)
    # add features from the preprocessed img (through a frozen cnn)
    train, test = merge_img_fcnn(img_dir, train, test)
        
    def add_rn(x):
        '''
        @todo should be a transformer
        :param x:
        '''
        x.loc[:, "rn"] = sample(range(10000000), x.shape[0])
        return x
    train = add_rn(train)
    test = add_rn(test)
    
    return train, test

def get_from_pkl(pathToAll, v = "getTrainTest2_meta_img_rn"):
    '''
    earn 1.5sec of your life * often
    :param pathToAll:
    :param v:
    '''
    full_precomputed_dir = pathToAll + "/preprocessed/full_precomputed/"
    pkl_file = full_precomputed_dir + v + ".pkl"
    if v == "getTrainTest2_meta_img_rn":
        if not isfile(pkl_file):
            with open(pkl_file, 'wb') as file:
                train, test = getTrainTest2_meta_img_rn(pathToAll)
                dump([train, test], file)
        else:
            with open(pkl_file, 'rb') as file:
                data = load(file)
                train = data[0]
                test = data[1]
    else:
        raise "@todo"
    return train, test

def quadratic_cohen_kappa(y_true, y_pred):
        return metrics.cohen_kappa_score(y_true, y_pred, weights = "quadratic")

def check_generalization(pipe, metric, X, y, test_size = 0.2, dishonnest_validation_mlp = False):
    '''
    Check for bad generalization to avoid overfit of a pipe param
    :param pipe:
    :param metric: scoring method 
    :param X: train and test
    :param y: train and test
    :warning :param dishonnest if True small leakage 
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
#     print([i[0] for i in pipe.steps])
#     print([i[0] for i in pipe.named_steps.u_prep.transformer_list])
   
    if dishonnest_validation_mlp:
        assert hasattr(pipe.named_steps.clf, "validation"), "no attribute validation, pipe's clf is no instance of CustomNNCategorical"
        prep_val_pipe = deepcopy(pipe)
        prep_val_pipe.steps = prep_val_pipe.steps[:-1] # only keep the preprocessing
        prep_val_pipe = prep_val_pipe.fit(x_train) #, y_train
        x_val = prep_val_pipe.transform(x_test)
        pipe.named_steps.clf.validation = (x_val, y_test)
        
    pipe.steps = pipe.steps[:-1] + [("dim_print", DimPrinter())] + pipe.steps[-1:] # add a print of X dimension

    pipe.fit(x_train, y_train)
    pred_train = pipe.predict(x_train)
    pred_test = pipe.predict(x_test)    
    
    score_train = metric(pred_train, y_train)
    score_test = metric(pred_test, y_test)
    
    gen = {"score_train":score_train, "score_test":score_test, "fitted_pipe": pipe}
    return gen
    
     
def printCatGen(gen, hist = False, plotname = "NN", saving_file = None):
    '''
    print result of check_generalization especially for custom NN
    :param gen: check_generalization() output
    :param hist: plot history?
    '''
    if hasattr(gen["fitted_pipe"].named_steps["clf"], "history"):
        epoch = len(gen["fitted_pipe"].named_steps["clf"].history.history["loss"])
        print("Num of epoch ", epoch, "(-patience).")
    print("gen['score_train'])",gen["score_train"])
    print("gen['score_test'])",gen["score_test"])
    
    if hist & hasattr(gen["fitted_pipe"].named_steps["clf"], "plot_history"):
        clf = gen["fitted_pipe"].named_steps["clf"]
        plot = clf.plot_history(plotname = plotname, saving_file = saving_file)
        return plot

def fitPrintGS(grid_search, X, y, pipeline = None, parameters = None, verboseP = 3):
    '''
    fit the grid search with x and y and print info about it
    :param grid_search:
    :param X: 
    :param y: target for x
    :param pipeline: of the grid search
    :param parameters: of the grid search
    '''
#     if not pipeline is  None: print("pipeline:", [name for name, _ in pipeline.steps])
#     if not parameters is None: print("parameters:")
#     if not parameters is None: pprint(parameters)
    t0 = time()
#     try:
    grid_search.fit(X, y)
#     except Exception as e:
#         traceback.print_tb(sys.exc_info()[2])
#         print("ERROR:", e)
#         pdb.post_mortem()
    
    if verboseP >= 2:
        print(grid_search.cv_results_["params"])
        for i in range(0, grid_search.n_splits_):
            if verboseP >= 3:
                print(str(i) + " train", grid_search.cv_results_["split"+str(i)+"_train_score"])
            print(str(i) + " test ", grid_search.cv_results_["split"+str(i)+"_test_score"])
        print("done in %0.3fs" % (time() - t0))
                
    if verboseP >= 1:
        print("Best score: %0.3f" % grid_search.best_score_)
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

def fitPrintPipe(pipe, X, y, scoring, cv, n_jobs, verbose=1, parameters = None, verboseP = 4):
    '''
    
    :param pipe:
    :param X:
    :param y:
    :param scoring:
    :param cv:
    :param n_jobs:
    :param verbose:
        - 0 nothing
        - 1 nb fit and fold + total time
        - 2 time for each
        - 3 score and time for each
    :param parameters:
    :param verboseP:
         - 0 nothing
         - 1 best score and avg
         - 2 all test score
         - 3 all train and test
         - 4 print X dim before clf
    '''
    if verboseP>=4:
        pipe.steps = pipe.steps[:-1] + [("dim_print", DimPrinter())] + pipe.steps[-1:] # add a print of X dimension
    grid_search = GridSearchCV(pipe, parameters, scoring = scoring, cv=cv,n_jobs=n_jobs,
                            verbose=verbose, return_train_score = True)                              
    fitPrintGS(grid_search, X = X, y = y, verboseP = verboseP,
                pipeline = pipe, parameters = parameters)
    