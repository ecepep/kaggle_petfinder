'''
Created on Jan 21, 2019

@author: ppc

list of pipelines to be tested in main_pipe

@todo: 
- meta_tfid features number = 9. wrong param of tfidf or unrelevant preprocessing/data?
- add rescuer to description (might be skipped by tf idf)
- include metadata label scores
- des_rescu_pipe
- could make a transformer isSmg(): 1 or 0 (breed2 = 0  means pure race for instance, state =? means kuala lumpur)
could feed well chosen breed to isSmg 

'''
from classification.transformer import PipeLabelEncoder, DataFrameSelector, InferNA, ToSparse, FnanToStr,\
    AsType, PipeOneHotEncoder, StringConcat, Ravel, Multiplier, DimPrinter,\
    ColorBreedOH
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
    
    :warning has to used pipe.steps and not pipe.named_step
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

breed = ['Breed1', 'Breed2']
color = ['Color1', 'Color2', 'Color3']

low_dim_only = breed + color + ['State']
nominal_features = low_dim_only + ['RescuerID']

text_features = ["Name", "Description"]
not_a_feat = ["AdoptionSpeed", "PetID"]

# metadata preprocesssed features :see preprocessed/metadata.py
expected_len = 10
meta_labels = ["label" + str(i) for i in range(0, expected_len)]
meta_label_scores = ["label_score" + str(i) for i in range(0, expected_len)]

# feat set with NAs at getTrainTest2, NAs to infer as mean of col
feat_with_nas = ordinal_features + binary_features

# on all train
# ColorBreedOH for breed  => 189 
# PipeOneHotEncoder for breed => 312
# PipeOneHotEncoder for state => 15
# PipeOneHotEncoder for RescuerID => 5596

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
            ('sel_nom', DataFrameSelector(low_dim_only)),
            ('encoder', PipeLabelEncoder(silent=True)),
            ('astype', AsType(astype = "float32")),
            ("sparse", ToSparse())
        ])

# one hot encoding of nominal_features
colors_oh = Pipeline([
            ('sel_nom', DataFrameSelector(color)),
            ('encoder', ColorBreedOH([1,0.6,0.3], silent=True)),
            ('astype', AsType(astype = "float64"))
        ])
breeds_oh = Pipeline([
            ('sel_nom', DataFrameSelector(breed)),
            ('encoder', ColorBreedOH([1,1], silent=True)),
            ('astype', AsType(astype = "float64"))
        ])
# low_dim_nom_pipe_oh_sparse = Pipeline([
#             ('nom_pipe_oh', low_dim_nom_pipe_oh),
#             ("sparse", ToSparse())
#         ])
state_oh = Pipeline([
            ('sel_nom', DataFrameSelector(["State"])),
            ('encoder', PipeOneHotEncoder(silent=True)),
            ('astype', AsType(astype = "float64")),
        ])

low_dim_nom_pipe_oh= Pipeline([
            ("ulowdim", FeatureUnion([
                ("colors", colors_oh),
                ("breed", breeds_oh),
                ("state", state_oh)
                ]))
        ])

rm_state_oh= Pipeline([
            ("ulowdim", FeatureUnion([
                ("colors", colors_oh),
                ("breed", breeds_oh)
                ]))
        ])
rm_breed_oh= Pipeline([
            ("ulowdim", FeatureUnion([
                ("colors", colors_oh),
                ("state", state_oh)
                ]))
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
    des_pipe_svd_v2, "SVD", ('SVD', TruncatedSVD(n_components=100)) )

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


