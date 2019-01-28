'''
Created on Jan 14, 2019

@author: ppc

main_pipe allow to run and assess the different pipelines


@todo 
- make a human readable output to save result of previously executed run with param

- ask on stacko comprehension list func scope
- @see bench and tf_test todos
- dir paths' hard coded
     
'''

from classification.util import getTrainTest2, quadratic_cohen_kappa_score,\
 check_generalization, printCatGen, fitPrintGS, fitPrintPipe

import sys

from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection._split import ShuffleSplit
from sklearn.model_selection._search import GridSearchCV

from multiprocessing import cpu_count
import warnings
import copy


# pd.options.mode.chained_assignment = 'raise'

# def custom_warn(*args, **kwargs):
#     raise "jkjkj" # for traceback
#  
# warnings.warn =  custom_warn

#################################################################"
#definition of all pipelines
from classification.pipelines import *
from preprocessed.metadata import merge_metadata
#################################################################"

pathToAll = "../all" # path to dataset dir
meta_dir = pathToAll + "/preprocessed/metadata_label/"
# read train, test csvs, set unknown to NA and shuffle
train, test = getTrainTest2(pathToAll)
# add features preprocessed from metadata dir :see preprocessed/prep_metadata
train, test = merge_metadata(meta_dir, train, test)

if __name__ == "__main__":
##############################################################################
# random forest
##############################################################################
    parameters = {
#          'clf__n_estimators': (200, 350), # high influence. 100: kappa ==> 0.340, 200 ==> 0.386, 200>300
# ValueError: n_components must be < n_features; got 140 >= 124
#         'u_prep__des_pipe_svd__SVD__n_components': (15, 25),
#         'u_prep__des_pipe__tfid_vect__max_df': (0.7, 0.743, 0.775),
#         'clf__min_samples_leaf': (1, 5, 10, 30,) # . 1 lower better
#         "clf__max_features":("sqrt", "log2", 2,3,4,5,6) # auto = sqrt, seems fine
    }
    # multiprocessing requires the fork to happen in a __main__ protected
    n_cpu = cpu_count()-1 #  1   cpu_count()-1
#     n_cpu = 1
    if n_cpu == 1: print("debug no multithread")
    else: print("using multithreading, bad for debug")
    cv_gs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=None)
    cv_gs = 3 # strat cross-val

    qwk_scorer = make_scorer(quadratic_cohen_kappa_score)
    
#     print("pipe_rdf_oh____________________________________________________")
#     fitPrintPipe(pipe_rdf_oh, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
#     print("pipe_rdf_des_svd____________________________________________________")
#     fitPrintPipe(pipe_rdf_des_svd, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)

#     print("pipe_rdf_des_svd_meta____________________________________________________")
#     fitPrintPipe(pipe_rdf_des_svd_meta, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)

    
#     print("pipe_rdf____________________________________________________")
#     fitPrintPipe(pipe_rdf, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
#     print("pipe_rdf_low_dim_only____________________________________________________")
#     fitPrintPipe(pipe_rdf_low_dim_only, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
#     print("pipe_rdf_des____________________________________________________")
#     fitPrintPipe(pipe_rdf_des, X = train, y = train["AdoptionSpeed"], 
#                  scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)


##############################################################################
# Custom MLP categorical_crossentropy and ordered with binary crossentropy
#############################################################################
    res_path = None
    res_path = pathToAll + "/result/mlp"
    if res_path:
        print("res_path", res_path)
        sys.stdout = open(res_path, 'w') # console output to file
    
    n_cpu = 1 # mutlithreaded already implemented through keras
    parameters = {}
    print("pipe_mlp ____________________________________________________")
    fitPrintPipe(pipe_mlp, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_low_dim_only ____________________________________________________")
    fitPrintPipe(pipe_mlp_low_dim_only, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_oh ____________________________________________________")
    fitPrintPipe(pipe_mlp_oh, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_oh_deeper ____________________________________________________")
    fitPrintPipe(pipe_mlp_oh_wider, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_oh_des ___________________________________________________")
    fitPrintPipe(pipe_mlp_oh_des, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_oh_des_svd ____________________________________________________")
    fitPrintPipe(pipe_mlp_oh_des_svd, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
    print("pipe_mlp_oh_des_svd_meta ____________________________________________________")
    fitPrintPipe(pipe_mlp_oh_des_svd_meta, X = train, y = train["AdoptionSpeed"], 
                 scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)

##############################################################################
# check generalization 
##############################################################################
#     print("rdf")
#     gen = check_generalization(pipe_rdf, metric = quadratic_cohen_kappa_score, 
#                          X = train.copy(), y = train["AdoptionSpeed"])
#     printCatGen(gen)
#     print("categorical")
#     gen = check_generalization(pipe_mlp, metric = quadratic_cohen_kappa_score, 
#                          X = train.copy(), y = train["AdoptionSpeed"])
#     printCatGen(gen)
#     print("ordered")
#     gen = check_generalization(pipe_mlp_ordered, metric = quadratic_cohen_kappa_score, 
#                          X = train.copy(), y = train["AdoptionSpeed"])
#     printCatGen(gen)

############################################################################"
# run result and conclusions
############################################################################"

"""
res for several != mlp in cv = 3, svd =20
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001]
    ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,5),\
strip_accents='ascii', analyzer= "word", stop_words = None, norm = "l1", use_idf = True)),

pipe_mlp ____________________________________________________
done in 2467.991s
Best score: 0.317

pipe_mlp_low_dim_only ____________________________________________________
done in 1587.536s
Best score: 0.209

pipe_mlp_oh ____________________________________________________
done in 4386.147s
Best score: 0.287

pipe_mlp_oh_deeper ____________________________________________________
done in 4454.292s
Best score: 0.273

pipe_mlp_oh_des ___________________________________________________
done in 6909.742s
Best score: 0.294

pipe_mlp_oh_des_svd ____________________________________________________
done in 5689.618s
Best score: 0.288

pipe_mlp_oh_des_svd_meta ____________________________________________________
done in 5945.231s
Best score: 0.295

pipe_mlp 317 > pipe_mlp_oh_des_svd_meta 295 ~ pipe_mlp_oh_des 294 >
pipe_mlp_oh_des_svd 288 ~ pipe_mlp_oh 287 > pipe_mlp_oh_deeper 273 > pipe_mlp_low_dim_only 0.209

Conclusion:
pipe_mlp_oh > pipe_mlp_low_dim_only => for low dim, oh still works better with mlp
pipe_mlp_oh 287 > pipe_mlp_oh_deeper 273 => wider mlp can't train
pipe_mlp_oh_des 294 > pipe_mlp_oh_des_svd 288 => svd still worse 
pipe_mlp_oh_des_svd_meta 295 ~ pipe_mlp_oh_des 294 => meta is maybe slightly intersting or NN lesser affected by a few more dimension 
pipe_mlp >> everything: always dim issues and/or importance of high dim features state and rescuer 
@todo add labelencoded high dim as a features and compare to pipe_mlp
@todo always print epoch for mlp
@todo test if cv = 1 is precise enough (computation time)
@todo compare each to rdf 
"""

"""
n_feat:
- tf-ifd for des ~= 128
- tf-idf for meta ~= 9
"""

"""
pipe_rdf_des_svd____________________________________________________
'clf__n_estimators': (200, 350),
'u_prep__des_pipe_svd__SVD__n_components': (40, 80, 100, 120)}
done in 1925.132s
Best score: 0.338
    clf__n_estimators: 350
    u_prep__des_pipe_svd__SVD__n_components: 40

2nd run:
'u_prep__des_pipe_svd__SVD__n_components': (10, 20, 30, 50)}
Best score: 0.345
    u_prep__des_pipe_svd__SVD__n_components: 20
3rd run:
'u_prep__des_pipe_svd__SVD__n_components': (15, 25)}
Best score: 0.349
    u_prep__des_pipe_svd__SVD__n_components: 15
4th run: n_components: 10, score : 0.339

pipe_rdf_des > pipe_rdf > pipe_svd (ncomponent = 15)>~ pipe_svd (ncomponent = 20) > pipe_svd (ncomponent = 10 or 30, 40 ... 100) ???? No conclusion?
pb with svd is just tuning. tf-idf has ~~124 features hence, svd components need to be reduced to be worth
still worse than des without svd, or dropping des BUT n_component 10 < 20??

"""
 
"""
running on several rdf pipes with cv = 3 and param n_estimators = 200 or 300, svd n component = 100 (wrong)
pipe_rdf_oh____________________________________________________
done in 79.987s
Best score: 0.291
Best parameters set:
    clf__n_estimators: 200
pipe_rdf_des_svd____________________________________________________
done in 731.013s
Best score: 0.316
    clf__n_estimators: 350
pipe_rdf_des_svd_meta____________________________________________________
done in 598.257s
Best score: 0.294
    clf__n_estimators: 200
pipe_rdf____________________________________________________
done in 42.563s
Best score: 0.352
    clf__n_estimators: 350
pipe_rdf_low_dim_only____________________________________________________
done in 207.250s
Best score: 0.307
    clf__n_estimators: 200
pipe_rdf_des____________________________________________________
done in 360.286s
Best score: 0.373
    clf__n_estimators: 350
 
____________   
conclusion:
pipe_rdf_des 373 > pipe_rdf 352 > pipe_rdf_des_svd 316 > pipe_rdf_low_dim_only 307 > pipe_rdf_des_svd_meta 294 > pipe_rdf_oh 291
know high dim?
true: pipe_rdf, pipe_rdf_des_svd, pipe_rdf_des_svd_meta, pipe_rdf_des
false: pipe_rdf_low_dim_only, pipe_rdf_oh

pipe_rdf_low_dim_only >  pipe_rdf_oh: for trees o-h lower scores du to dimension issues.
problem of behaviour in higher dimension overall => same info with more dim lower result significatively
pipe_rdf >> pipe_rdf_low_dim_only: high dimension feats help a lot if as label!!!
pipe_rdf_des > pipe_rdf: des very usefull but shouldn't it curse dim? (or "only" ~~124 extra dim?)
pipe_rdf_des > pipe_rdf_des_svd: SVD actually worsening problem => normal with ncomponent = 100 see comment above
"""

"""
pipe_rdf_oh
Best score: 0.289pipe_rdf_oh

pipe_mlp
Fitting 3 folds for each of 1 candidates, totalling 3 fits : 41.3min finished
Best score: 0.317pipe_ml

pipe_mlp_oh
Best score: 0.280
"""



 
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

# pipe_rdf_des_svd_meta = pipe_rdf_des_svd_meta.fit(X = train, y = train["AdoptionSpeed"])
# meta_tfid = pipe_rdf_des_svd_meta.named_steps.u_prep.transformer_list[3][1].named_steps.tfid_vect
# print( "nb feat meta_tfid:", meta_tfid.get_feature_names() )
# print( "nb feat meta_tfid:", len(meta_tfid.get_feature_names()) )
# des_tfid = pipe_rdf_des_svd_meta.named_steps.u_prep.transformer_list[2][1].named_steps.des_pipe.named_steps.tfid_vect
# print( "nb feat des_tfid:", des_tfid.get_feature_names() )
# print( "nb feat des_tfid:", len(des_tfid.get_feature_names()) )

# output
# nb feat meta_tfid: ['cat', 'cat small', 'dog', 'dog dogbr', 'dog dogli', 'dogbr', 'dogbreed', 'dogli', 'small']
# nb feat meta_tfid: 9
# nb feat des_tfid: ['abandoned', 'able', 'active', 'adopt', 'adopted', 'adopter', 'adopters', 'adoption', 'adorable', 'age', 'ago', 'area', 'attention', 'away', 'beautiful', 'birth', 'black', 'born', 'boy', 'breed', 'bring', 'care', 'cat', 'cats', 'come', 'contact', 'currently', 'cute', 'day', 'dewormed', 'dog', 'dogs', 'don', 'eat', 'email', 'eyes', 'family', 'fee', 'female', 'food', 'forever', 'forever home', 'free', 'friend', 'friendly', 'fur', 'gave', 'girl', 'given', 'good', 'good home', 'happy', 'healthy', 'help', 'home', 'hope', 'house', 'interested', 'interested adopt', 'just', 'kind', 'kitten', 'kittens', 'know', 'left', 'let', 'life', 'like', 'litter', 'little', 'location', 'long', 'look', 'looking', 'love', 'lovely', 'loves', 'loving', 'loving home', 'make', 'male', 'manja', 'month', 'months', 'months old', 'mother', 'near', 'need', 'needs', 'neutered', 'neutering', 'new', 'new home', 'old', 'owner', 'people', 'pet', 'place', 'play', 'playful', 'pls', 'provide', 'puppies', 'puppy', 'ready', 'really', 'rescued', 'siblings', 'small', 'sms', 'soon', 'spay', 'spayed', 'stray', 'sweet', 'thank', 'thanks', 'time', 'toilet', 'trained', 'vaccinated', 'vaccination', 'vet', 'want', 'weeks', 'whatsapp', 'white', 'willing']
# nb feat des_tfid: 128