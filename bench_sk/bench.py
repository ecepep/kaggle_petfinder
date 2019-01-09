'''
Created on Jan 4, 2019

@author: ppc

This is just a bench

@todo
- try with gbt
- clean code with everything in a pipeline
- go to keras_test to use perceptron more extensively (optimized on kappa)
- separate preprocessing from analysis
- add more data: photo metadata, text, photo...
- FIND alternate to one hot encode, fight curse of dim: regularization, 
grouping color and breed, hashing tricks?
'''


"""
Other parameters you may want to look at are those controlling how big a tree can grow.
As mentioned above, averaging predictions from each tree counteracts overfitting, so usually one wants biggish trees.

One such parameter is min. samples per leaf. In scikit-learn’s RF,
it’s value is one by default. Sometimes you can try increasing this value a little bit
to get smaller trees and less overfitting. This CoverType benchmark overdoes it, 
going from 1 to 13 at once. Try 2 or 3 first.

https://arxiv.org/abs/1506.03410
"""
from bench_sk.preprocessing import *

 
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn.ensemble.forest import RandomForestClassifier
import sklearn.neural_network as neural_network
from sklearn.pipeline import Pipeline

pathToAll = "../all/"

trainPath = pathToAll + "/train.csv"
print("trainPath ", trainPath)
testPath = pathToAll + "/test/test.csv"
print("testPath ", testPath)

# num & ordinal
numeric_features = ['Age', 'MaturitySize', 'FurLength', 'Health', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
nominal_features = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'Vaccinated',
                     'Dewormed', 'Sterilized', 'State', 'RescuerID']

# read csv and set some to na
train = readBaseCSV(trainPath)
test = readBaseCSV(testPath)

# infer na to mean of value (even for unordered value because they are all binary)
train, test = inferNa(train, test, 
                      infer = {"Health" : "mean", 
                                "MaturitySize" : "mean",
                                "FurLength" : "mean", 
                                "Gender" : "mean",
                                "Vaccinated" : "mean",
                                "Dewormed" : "mean",
                                "Sterilized" : "mean"})

toEncode = ['State', 'RescuerID']
train, test = encodeLabel(train, test,  toEncode)

# can probably drop binary (yes, no, na) features from the one_hot transformer 
# has been set (after standardization) to yes = 0, no = 1, na = mean (~0.5)
toFlatten = np.setdiff1d(nominal_features, ["Sterilized", "Dewormed", "Vaccinated", "Gender"])
train, test = flattenLabel(train, test, toFlatten, drop = 4)
 
def quadratic_cohen_kappa_score(*arg):
    return metrics.cohen_kappa_score(*arg, weights = "quadratic")
 
qck_scorer = metrics.make_scorer(quadratic_cohen_kappa_score)

approFeatN = train.shape[1]
print("final shape", train.shape[1])

# HashingEncoder
# binaryencoder

# standardize data before mlp
# @todo get rid of warn
pipe = Pipeline([("scaler", preprocessing.StandardScaler()), 
    ("NNmlp", neural_network.MLPClassifier(
        hidden_layer_sizes=(100, ), 
        activation="relu", 
        solver="adam", 
        alpha=0.0001, 
        batch_size="auto", 
        learning_rate="constant", 
        learning_rate_init=0.001, 
        power_t=0.5, max_iter=800, 
        shuffle=True, random_state=None,
        tol=0.0001, verbose=False, 
        warm_start=False, momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, validation_fraction=0.1,
        beta_1=0.9, beta_2=0.999,
        epsilon=1e-08, n_iter_no_change=10) )])
resultMLP = model_selection.cross_validate(pipe,
                                        X = train.drop(["AdoptionSpeed", "PetID"], axis = 1), 
                                        y = train["AdoptionSpeed"], 
                                        scoring= qck_scorer,
                                        cv= 3, n_jobs=None, verbose=0)
print("resultMLP", resultMLP)

resultRDF = model_selection.cross_validate(RandomForestClassifier(),
                                        X = train.drop(["AdoptionSpeed", "PetID"], axis = 1), 
                                        y=train["AdoptionSpeed"], 
                                        scoring= qck_scorer,
                                        cv= 3, n_jobs=None, verbose=0)
print("resultRDF", resultRDF)
# ~~ 0.29 with label encoding

"""
resultMLP {'fit_time': array([14.09949327, 23.89691162, 26.43237019]), 'score_time': array([0.01697373, 0.01588368, 0.01575494]), 'test_score': array([0.25520967, 0.2723275 , 0.28131855]), 'train_score': array([0.35918377, 0.35014591, 0.35509816])}
resultRDF {'fit_time': array([0.13120365, 0.11526203, 0.1050806]), 'score_time': array([0.02887034, 0.02511621, 0.02473092]), 'test_score': array([0.21194344, 0.22549286, 0.2199938 ]), 'train_score': array([0.82416418, 0.81690801, 0.8195138 ])}
conclusion: 
rdf lost 0.29 -> 0.18 when labelencoding was replace by one-hot. very likely to be curse of dimension (to few data)
mlp also around 0.19 with one hot
Though when removing one hot (aka high dim) mlp >> rdf
"""