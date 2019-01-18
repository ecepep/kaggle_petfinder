'''
Created on Jan 14, 2019

@author: ppc

@todo 
- TruncatedSCD
- implement method ordered
- extra data
- data augmentation from ggle
- @see bench and tf_test todos
     
'''

from bench_sk.preprocessing import *
from classification.util import *
from classification.transformer import *
from classification.custom_nn_base import CustomNNBase
from classification.custom_nn_categorical import CustomNNCategorical

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection._split import ShuffleSplit
import tensorflow as tf

import copy
from multiprocessing import cpu_count

import warnings
from classification.custom_nn_ordered import CustomNNordered
from sklearn.model_selection._search import GridSearchCV

# pd.options.mode.chained_assignment = 'raise'

# def custom_warn(*args, **kwargs):
#     raise "jkjkj" # for traceback
#  
# warnings.warn =  custom_warn



pathToAll = "../all" # path to dataset dir
# read train, test csvs, set unknown to NA and shuffle
train, test = getTrainTest2(pathToAll)

# csv features' type
numeric_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
ordinal_features = ['MaturitySize', 'FurLength', 'Health']
nominal_features = ['Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'State', 'RescuerID']
binary_features = ['Gender', 'Vaccinated', 'Dewormed', 'Sterilized']
text_features = ["Name", "Description"]

not_a_feat = ["AdoptionSpeed", "PetID"]
# feat set with NAs at getTrainTest2, NAs to infer as mean of col
feat_with_nas = ordinal_features + binary_features


##############################################################################
# base for pipes
##############################################################################
num_pipe = Pipeline([
            ('sel_num', DataFrameSelector(numeric_features + binary_features + ordinal_features, dtype = 'float32')),
            ("scaler", StandardScaler())
        ])

nom_pipe_label_encode = Pipeline([
            ('sel_nom', DataFrameSelector(nominal_features)),
            ('encoder', PipeLabelEncoder(silent=True))
        ])

##############################################################################
# pipes for random forest
##############################################################################
pipe_rdf = Pipeline([
    ('infer_na_mean', inferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])
    
##############################################################################
# pipes for mlp
##############################################################################
print("nom_pipe_label_encode_scale :/")
nom_pipe_label_encode_scale = Pipeline([
            ('sel_nom', DataFrameSelector(nominal_features)),
            ('encoder', PipeLabelEncoder(silent=True, astype = "float64")),
# loosy way of running mlp with nominal without burning dim for fairer compare with trees
            ("scaler", StandardScaler())            
        ])

epoch = 500
pipe_mlp = Pipeline([
    ('infer_na_mean', inferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode_scale', nom_pipe_label_encode_scale)
    ])),
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))
])

pipe_mlp_ordered = copy.deepcopy(pipe_mlp)
pipe_mlp_ordered.named_steps.clf = CustomNNordered(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch)

if __name__ == "__main__":
##############################################################################
# random forest
##############################################################################
    parameters = {
#         'clf__n_estimators': (100, 200, 300, 400), # high influence. 100: kappa ==> 0.340, 200 ==> 0.386, 200>300
    #     'clf__min_samples_leaf': (1, 5, 10, 30,) # . 1 lower better
        "clf__max_features":("sqrt", "log2", 2,3,4,5,6) 
    }
    # multiprocessing requires the fork to happen in a __main__ protected
    n_cpu =  cpu_count() 
    if n_cpu == 2: print("debug no multithread")
    cv_gs = 3 # strat cross-val
    cv_gs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=None)

    qwk_scorer = metrics.make_scorer(quadratic_cohen_kappa_score)
    grid_search = GridSearchCV(pipe_rdf, parameters, scoring = qwk_scorer, cv=cv_gs,
                               n_jobs=n_cpu-1, verbose=1)
    fitPrintGS(grid_search, X = train.copy(), y = train["AdoptionSpeed"],
                pipeline = pipe_rdf, parameters = parameters)
     
    
##############################################################################
# Custom MLP categorical_crossentropy
##############################################################################
    n_cpu = 2 # mutlithreaded alreay implement through keras (btw: 1 = 2-1)
#     grid_search = GridSearchCV(pipe_rdf, parameters, scoring = qwk_scorer, cv=cv_gs,
#                                n_jobs=n_cpu-1, verbose=1)
#     fitPrintGS(grid_search, X = train.copy(), y = train["AdoptionSpeed"],
#             pipeline = pipe_rdf, parameters = parameters)
#     grid_search = GridSearchCV(pipe_rdf, parameters, scoring = qwk_scorer, cv=cv_gs,
#                                n_jobs=n_cpu-1, verbose=1)
#     fitPrintGS(grid_search, X = train.copy(), y = train["AdoptionSpeed"],
#             pipeline = pipe_rdf, parameters = parameters)

        
    print("rdf")
    gen = check_generalization(pipe_rdf, metric = quadratic_cohen_kappa_score, 
                         X = train.copy(), y = train["AdoptionSpeed"])
    printCatGen(gen)
    print("categorical")
    gen = check_generalization(pipe_mlp, metric = quadratic_cohen_kappa_score, 
                         X = train.copy(), y = train["AdoptionSpeed"])
    printCatGen(gen)
    print("ordered")
    gen = check_generalization(pipe_mlp_ordered, metric = quadratic_cohen_kappa_score, 
                         X = train.copy(), y = train["AdoptionSpeed"])
    printCatGen(gen)
 
"""
easy overfit with nom_pipe_label_encode_scale feat (too high variance)
cat
(dropout = [0.3,0.2,0.1]
gen['score_train']) 0.6290646090324274
gen['score_test']) 0.32729647178868726
 
cat
dropout = [0.2,0.1], 
reg = [0.0005,0.0005,0.0005]
gen['score_train']) 0.6416221164691353
gen['score_test']) 0.26817974398357936
 
cat
dropout = [0.2,0.1], 
reg = [0.01,0.005,0.0005,0.0005]
gen['score_train']) 0.45056943772778246
gen['score_test']) 0.2954824007300981
 
cat
dropout = [0.2,0.1], 
reg = [0.01,0.005,0.001,0.001]
gen['score_train']) 0.3900815959013235
gen['score_test']) 0.32012826618688706
 
cat
dropout = [0.2,0.1], 
reg = [0.01,0.005,0.005,0.005]
gen['score_train']) 0.33
gen['score_test']) 0.28
 
cat
dropout = [0.3,0.2,0.1], 
reg = [0.01,0.005,0.001]
gen['score_train']) 0.40597548911768644
gen['score_test']) 0.3154058123573781
 
cat
dropout = [0.3,0.2,0.1], 
reg = [0.01,0.005,0.001]
gen['score_train']) 0.40597548911768644
gen['score_test']) 0.3154058123573781
 
ordered
gen['score_train']) 0.3844140591568229
gen['score_test']) 0.3184767641302274

categorical
hidden = [300, 200, 100, 50, 20],
dropout = [0.3,0.2,0.1], 
reg = [0.01,0.005,0.001]
2019-01-18 11:51:19.681611: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2019-01-18 11:51:19.686810: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
after  203 epochs.
gen['score_train']) 0.4359417358581271
gen['score_test']) 0.33416616155622014
ordered
after  201 epochs.
gen['score_train']) 0.4375396917454981
gen['score_test']) 0.35719358709416804
 
"""
 
     