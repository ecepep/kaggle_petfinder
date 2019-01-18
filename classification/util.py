'''
Created on Jan 14, 2019

@author: ppc
'''

from bench_sk.preprocessing import *
import sklearn.metrics as metrics

from pprint import pprint
from time import time
from sklearn.model_selection._split import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


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

def quadratic_cohen_kappa_score(*arg):
    return metrics.cohen_kappa_score(*arg, weights = "quadratic")

def check_generalization(pipe, metric, X, y, test_size = 0.2):
    '''
    Check for bad generalization to avoid overfit of a pipe param
    :param pipe:
    :param scoring:
    :param X: train and test
    :param y: train and test
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    pipe.fit(x_train, y_train)
    pred_train = pipe.predict(x_train)
    pred_test = pipe.predict(x_test)
    
    score_train = metric(pred_train, y_train)
    score_test = metric(pred_test, y_test)
    return {"score_train":score_train, "score_test":score_test, "fitted_pipe": pipe}
    
     
def printCatGen(gen, hist = False):
    '''
    print result of check_generalization especially for custom NN
    :param gen: check_generalization() output
    :param hist: plot history?
    '''
    if hist & hasattr(gen["fitted_pipe"].named_steps["clf"], "plot_history"):
        gen["fitted_pipe"].named_steps["clf"].plot_history()
    if hasattr(gen["fitted_pipe"].named_steps["clf"], "history"):
        epoch = len(gen["fitted_pipe"].named_steps["clf"].history.history["loss"])
        print("after ", epoch, "epochs.")
    print("gen['score_train'])",gen["score_train"])
    print("gen['score_test'])",gen["score_test"])
    

def fitPrintGS(grid_search, X, y, pipeline = None, parameters = None):
    '''
    fit the grid search with x and y and print info about it
    :param grid_search:
    :param X: 
    :param y: target for x
    :param pipeline: of the grid search
    :param parameters: of the grid search
    '''
    print("Performing grid search...")
    if not pipeline is  None: print("pipeline:", [name for name, _ in pipeline.steps])
    if not parameters is None: print("parameters:")
    if not parameters is None: pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

