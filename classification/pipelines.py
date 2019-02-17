'''
Created on Jan 21, 2019

@author: ppc

list of pipelines to be tested in main_pipe

@todo: 
- meta_tfid features number = 9. wrong param of tfidf or unrelevant preprocessing/data?
- feature for state (nb inhabitant?)
- add rescuer to description (might be skipped by tf idf)
- add colors and breeds to each other with weights
- include metadata label scores
- des_rescu_pipe

'''
from classification.transformer import PipeLabelEncoder, DataFrameSelector, InferNA, ToSparse, FnanToStr,\
    AsType, PipeOneHotEncoder, StringConcat, Ravel, Multiplier, DimPrinter
from classification.custom_nn_base import CustomNNBase
from classification.custom_nn_categorical import CustomNNCategorical

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier,\
    RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from classification.custom_nn_ordered import CustomNNordered
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD

from tensorflow.nn import relu
from keras.callbacks import EarlyStopping

import copy
import math
from sklearn.linear_model.base import LinearRegression
from NN_loss.ordinal_categorical_crossentropy import lossOCC, lossOCCQuadratic

# @todo ask stack ???
# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("reg", LinearRegression()),
#     ])
# pipe.named_steps.reg = RandomForestRegressor()
# print(pipe) # won't assign
# pipe.steps[1] =  ("reg", RandomForestRegressor())
# print(pipe) # will assign
def replace_step(pipe, step_name, new_step):
    '''
    :param pipe: original pipe
    :param step_name: as string :ex "clf"
    :param new_step: the usual tupple :ex ("clf", RandomForest())
    :return a new pipe where the step as been replace
    '''
    for step in range(0, len(pipe.steps)):
        if pipe.steps[step][0] ==  step_name:
            new_pipe = copy.deepcopy(pipe)
            new_pipe.steps[step] = new_step
            return  new_pipe
    raise Exception("step not found:"+step_name)

## csv features' type
#feat used in num pipe
numeric_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
ordinal_features = ['MaturitySize', 'FurLength', 'Health']
binary_features = ['Gender', 'Vaccinated', 'Dewormed', 'Sterilized']

print("separation low high to be questionned for breed and state, I wonder if they ain't the same num of label. Incoherence? @todo")
breed = ['Breed1', 'Breed2']
color = ['Color1', 'Color2', 'Color3']
h_dim_nom = [ 'State', 'RescuerID']
low_dim_nom = breed + color
nominal_features = breed + color + h_dim_nom

text_features = ["Name", "Description"]
not_a_feat = ["AdoptionSpeed", "PetID"]

# metadata preprocesssed features :see preprocessed/metadata.py
expected_len = 10
meta_labels = ["label" + str(i) for i in range(0, expected_len)]
meta_label_scores = ["label_score" + str(i) for i in range(0, expected_len)]

# feat set with NAs at getTrainTest2, NAs to infer as mean of col
feat_with_nas = ordinal_features + binary_features


##############################################################################
# base for pipes
##############################################################################
#X dim print (8995, 12)
num_pipe = Pipeline([
            ('sel_num', DataFrameSelector(numeric_features + binary_features + ordinal_features, dtype = 'float32')),
            ("scaler", StandardScaler())
        ])

pipe_real_num_only = Pipeline([
            ('sel_num', DataFrameSelector(numeric_features, dtype = 'float32')),
            ("scaler", StandardScaler())
        ])
num_pipe_binary_only = Pipeline([
            ('sel_num', DataFrameSelector(binary_features, dtype = 'float32')),
            ("scaler", StandardScaler())
        ])

num_pipe_sparse = Pipeline([
            ('num_pipe', num_pipe),
            ("sparse", ToSparse())
        ])

nom_pipe_label_encode = Pipeline([
            ('sel_nom', DataFrameSelector(nominal_features)),
            ('encoder', PipeLabelEncoder(silent=True)),
            ('astype', AsType(astype = "float32"))
        ])
nom_pipe_label_encode_sparse = Pipeline([
            ('nom_pipe_label_encode', nom_pipe_label_encode),
            ("sparse", ToSparse())
        ])

nom_pipe_label_encode_low_dim_scale = Pipeline([
            ('sel_nom', DataFrameSelector(low_dim_nom)),
            ('encoder', PipeLabelEncoder(silent=True)),
            ('astype', AsType(astype = "float32")),
            ("sparse", ToSparse())
        ])


h_dim_nom_pipe_label_encode = Pipeline([
            ('sel_nom', DataFrameSelector(h_dim_nom)),
            ('encoder', PipeLabelEncoder(silent=True)),
            ('astype', AsType(astype = "float32"))
        ])
h_dim_nom_pipe_label_encode_sparse = Pipeline([
            ('nom_pipe_label_encode', nom_pipe_label_encode),
            ("sparse", ToSparse())
        ])

# one hot encoding of nominal_features
low_dim_nom_pipe_oh = Pipeline([
            ('sel_nom', DataFrameSelector(low_dim_nom)),
            ('encoder', PipeOneHotEncoder(silent=True)),
            ('astype', AsType(astype = "float64"))
        ])
low_dim_nom_pipe_oh_sparse = Pipeline([
            ('nom_pipe_oh', low_dim_nom_pipe_oh),
            ("sparse", ToSparse())
        ])
state_oh = Pipeline([
            ('sel_nom', DataFrameSelector(["State"])),
            ('encoder', PipeOneHotEncoder(silent=True)),
            ('astype', AsType(astype = "float64")),
        ])


# alt to nom_pipe_oh
# loosy way of running mlp with nominal without burning dim for fairer compare with trees
nom_pipe_label_encode_scale = Pipeline([
            ('nom_pipe', nom_pipe_label_encode),
            ('astype', AsType(astype = "float64")),
            ("scaler", StandardScaler())            
        ])

print("des pipe, tfidf param set for mail classification @todo")
des_pipe = Pipeline([
            ('sel_num', DataFrameSelector(["Description"], ravel = True)),
            ('rm_nan', FnanToStr()),
            ("ravel", Ravel()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,4),\
strip_accents='ascii', analyzer= "word", stop_words='english', norm = "l1", use_idf = True))
        ])

# des_rescu_pipe = Pipeline([
#             ('sel_num', DataFrameSelector(["Description", "RescuerID"], ravel = True)),
#             add rescuer to description
#             ('rm_nan', FnanToStr()),
#             ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,4),\
# strip_accents='ascii', analyzer= "word", stop_words='english', use_idf = True, norm = None))
#         ])

des_pipe_svd = Pipeline([
            ('des_pipe', des_pipe),
            ('SVD', TruncatedSVD(n_components=20)) #ValueError: n_components must be < n_features; got 140 >= 124
        ])

des_pipe_for_svd = replace_step(
    des_pipe, 
    "tfid_vect", 
    ("tfid_vect", TfidfVectorizer(max_df= 0.95, min_df=0.005, ngram_range=(1,4),\
                                  strip_accents='ascii', analyzer= "word", stop_words='english', 
                                  norm = "l1", use_idf = True))
)
des_pipe_svd_v2 = Pipeline([
            ('des_pipe_for_svd', des_pipe_for_svd),
            ('SVD', TruncatedSVD(n_components=20))
        ])

des_pipe_svd_v3 = replace_step(
    des_pipe_svd_v2, 
    "SVD", 
    ('SVD', TruncatedSVD(n_components=100))
)

print("meta_label_simple_concat_pipe pipe, tfidf param set for mail classification @todo")
meta_label_simple_concat_pipe = Pipeline([
            ('sel_label', DataFrameSelector(meta_labels)),
            ('rm_nan', FnanToStr()),#supposed to be useless,
            ('concat_labels', StringConcat()),
            ("ravel", Ravel()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,5),\
strip_accents='ascii', analyzer= "word", stop_words = None, norm = "l1", use_idf = True)),
])

pipe_img = Pipeline([
    ("sel_imgf", DataFrameSelector("imgf_[0-9]+", regex = True)), 
    ("scaler", StandardScaler())
    ])

pipe_img_PCA = Pipeline([("sel_imgf", pipe_img),
                         ("PCA", PCA(n_components=30))])

n_extra = 10
extra_dim = Pipeline([ 
    ('extra_dim', FeatureUnion([
         ('sel_rn'+str(i), DataFrameSelector(["rn"], ravel = False))
    for i in range(0, n_extra)]))
    ])


##############################################################################
# pipes for random forest
##############################################################################

# most basic pipe for reference, gives decent result ~0.340
pipe_rdf = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])

pipe_rdf_extra_dim = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode),
        ("extra_dim", extra_dim),
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

# supposed to give 0
pipe_rn_cmp = Pipeline([
    ('sel_rn', DataFrameSelector(["rn"], ravel = False)),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])

pipe_rdf_img_only = pipe_rdf_extra_dim = Pipeline([
    ("pipe_img", pipe_img),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])

pipe_rdf_img_PCA_only = pipe_rdf_extra_dim = Pipeline([
    ("pipe_img_PCA", pipe_img_PCA),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

pipe_rdf_meta_only = pipe_rdf_extra_dim = Pipeline([
    ('meta_label_simple_concat_pipe', meta_label_simple_concat_pipe),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

# for comparison
pipe_rdf_real_num_only = pipe_rdf_extra_dim = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ("pipe_real_num_only", pipe_real_num_only),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 
# for comparison
pipe_rdf_binary_only = pipe_rdf_extra_dim = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ("num_pipe_binary_only", num_pipe_binary_only),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

# only the description
pipe_rdf_des_only = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('des_pipe', des_pipe),
    ('clf', RandomForestClassifier(n_estimators = 300)),
])



pipe_rdf_img = pipe_rdf_extra_dim = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode),
        ("pipe_img", pipe_img)
    ])),

    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

pipe_rdf_img_PCA = pipe_rdf_extra_dim = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode),
        ("pipe_img_PCA", pipe_img_PCA)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 

# add desccription features 
pipe_rdf_des = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ('nom_pipe_label_encode', nom_pipe_label_encode_sparse),
        ('des_pipe', des_pipe)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 300)),
])



pipe_rdf_des_svd = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ('nom_pipe_label_encode', nom_pipe_label_encode_sparse),
        ('des_pipe_svd', des_pipe_svd)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 300)),
])
pipe_rdf_des_svd_v2 = copy.deepcopy(pipe_rdf_des_svd)
pipe_rdf_des_svd_v2.named_steps.u_prep.transformer_list[2] = ('des_pipe_svd_v2', des_pipe_svd_v2)
pipe_rdf_des_svd_v3 = copy.deepcopy(pipe_rdf_des_svd)
pipe_rdf_des_svd_v3.named_steps.u_prep.transformer_list[2] = ('des_pipe_svd_v3', des_pipe_svd_v3)
 
pipe_rdf_des_svd_meta = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ('nom_pipe_label_encode', nom_pipe_label_encode_sparse),
        ('des_pipe_svd', des_pipe_svd),
        ('meta_label_simple_concat_pipe', meta_label_simple_concat_pipe)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])

# test one hot encode against label encoding
pipe_rdf_oh = copy.deepcopy(pipe_rdf)
pipe_rdf_oh.named_steps.u_prep.transformer_list[1] = ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh)
pipe_rdf_oh = replace_step(pipe_rdf_oh, "clf", ("clf", RandomForestClassifier(n_estimators = 300)) )

# test impact of state and rescuer as label on precision
pipe_rdf_low_dim_only = copy.deepcopy(pipe_rdf)
pipe_rdf_low_dim_only.named_steps.u_prep.transformer_list[1] = \
    ('nom_pipe_label_encode_low_dim_scale', nom_pipe_label_encode_low_dim_scale)
    
##############################################################################
# pipes for mlp
##############################################################################

epoch = 500 # 1, 500
if epoch == 1: print("epoch == 1; debugging for mlp")

# Removal of regularization in further model to allow low dim features to be put forward
# old basic does early stop on train accuracy (wrong metric, no val)
oldBasicNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [relu],
                                epoch = epoch,
                                cbEarly = EarlyStopping(
                                    monitor='acc', 
                                    min_delta=0.000010, patience=20, 
                                    verbose=0, mode='max', 
                                    restore_best_weights = True),
                                metrics = ["accuracy"])

# https://www.reddit.com/r/MachineLearning/comments/3oztvk/why_50_when_using_dropout/
# http://papers.nips.cc/paper/4878-understanding-dropout.pdf

basicNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])

# probably not of interest
# lowDoNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.1,0.1,0.1], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# DoButlastNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# input do inputna01NN > inputnaNN > input05NN > input08NN

# inputnaNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [None,0.5,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# input08NN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.8,0.5,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# inputna01NN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
raise 'add deeper but tiny'
lossOCCNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric", loss = lossOCC,
                                metrics = ['cohen_kappa'])
lossOCCQuadraticNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric", loss = lossOCCQuadratic,
                                metrics = ['cohen_kappa'])

simplerNN = CustomNNCategorical(hidden = [200, 100, 50, 20], dropout = [0.1,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])
shallowerNN = CustomNNCategorical(hidden = [400, 200, 200], dropout = [0.1,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])
shallower2NN = CustomNNCategorical(hidden = [200, 200], dropout = [0.1,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])
shallowerWiderNN = CustomNNCategorical(hidden = [400, 400, 400], dropout = [0.1,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])
shallowerWider2NN = CustomNNCategorical(hidden = [500, 500], dropout = [0.1,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])

widerNN = CustomNNCategorical(hidden = [600, 300, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = epoch, cbEarly = "metric",
                                metrics = ['cohen_kappa'])



# DoStrongNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.5,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# DoWeakNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.1], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])

reguNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [],
                               reg = [0.005,0.005,0.005,0.005], h_act = [relu],
                               epoch = epoch, cbEarly = "metric",
                               metrics = ['cohen_kappa'])

orderedNN = CustomNNordered(
        hidden = [400, 200, 100, 50, 20],
        dropout = [0.5,0.5,0.5,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric",
        metrics = ['cohen_kappa'])

#for debug
dummyesNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1, 0.5,0.5,0.5,0.5], reg = [],
                               h_act = [relu], epoch = 300, cbEarly = EarlyStopping(patience=math.inf),
                                metrics = ['cohen_kappa'])
dummyesNN.metric_plot = 'cohen_kappa'

pipe_mlp = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode_scale', nom_pipe_label_encode_scale)
    ])),
    ('clf', copy.deepcopy(basicNN))
])


pipe_mlp_low_dim_only = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode_low_dim_scale', nom_pipe_label_encode_low_dim_scale)
    ])),
    ('clf', copy.deepcopy(basicNN))
])

pipe_mlp_oh = copy.deepcopy(pipe_mlp)
pipe_mlp_oh.named_steps.u_prep.transformer_list[1] = ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh)


pipe_mlp_oh_des_svd = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh),
        ('des_pipe_svd', des_pipe_svd),
    ])),
    ('clf', copy.deepcopy(basicNN))
])

pipe_mlp_oh_des = copy.deepcopy(pipe_mlp_oh_des_svd)
pipe_mlp_oh_des.named_steps.u_prep.transformer_list[2] = ('des_pipe', des_pipe)

pipe_mlp_oh_des_svd_meta = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh),
        ('des_pipe_svd', des_pipe_svd),
        ('meta_label_simple_concat_pipe', meta_label_simple_concat_pipe)
    ])),
    ('clf', copy.deepcopy(basicNN))
])

pipe_mlp_img_only = pipe_rdf_extra_dim = Pipeline([
    ("pipe_img", pipe_img),
    ('clf', copy.deepcopy(basicNN)),
]) 

pipe_mlp_oh_img = copy.deepcopy(pipe_mlp_oh)
pipe_mlp_oh_img.named_steps.u_prep.transformer_list.append(("pipe_img", pipe_img))

pipe_mlp_oh_img_PCA = copy.deepcopy(pipe_mlp_oh)
pipe_mlp_oh_img_PCA.named_steps.u_prep.transformer_list.append(("pipe_img_PCA", pipe_img_PCA))

pipe_mlp_oh_des_img_PCA = copy.deepcopy(pipe_mlp_oh_des)
pipe_mlp_oh_des_img_PCA.named_steps.u_prep.transformer_list.append(("pipe_img_PCA", pipe_img_PCA))


# pipe_mlp_doStrong_oh_des_img_PCA = replace_step(pipe_mlp_oh_des_img_PCA, "clf", ('clf', copy.deepcopy(DoStrongNN)) )
pipe_mlp_simpler_oh_des_img_PCA = replace_step(pipe_mlp_oh_des_img_PCA, "clf", ('clf', copy.deepcopy(simplerNN)) )
         
# pipe for different NN param
# pipe_mlp_DoStrongNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(DoStrongNN)) )
# pipe_mlp_DoWeakNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(DoWeakNN)) )
pipe_mlp_reguNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(reguNN)) )
pipe_mlp_orderedNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(orderedNN)) )
# pipe_mlp_oh_inputnaNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(inputnaNN)) )
# pipe_mlp_oh_inputna01NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(inputna01NN)) )
# pipe_mlp_oh_input08NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(input08NN)) )
# pipe_mlp_oh_DoButlastNN= replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(DoButlastNN)) )
# pipe_mlp_oh_lowDoNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(lowDoNN)) )

pipe_mlp_oh_lossOCCNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(lossOCCNN)) )
pipe_mlp_oh_lossOCCQuadraticNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(lossOCCQuadraticNN)) )

pipe_mlp_simpler_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(simplerNN)) )
pipe_mlp_oh_wider = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(widerNN)) )
pipe_mlp_oh_shallowerNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerNN)) )
pipe_mlp_oh_shallower2NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower2NN)) )
pipe_mlp_oh_shallowerWiderNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerWiderNN)) )
pipe_mlp_oh_shallowerWider2NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerWider2NN)) )

pipe_mlp_dummyesNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(dummyesNN)) )


