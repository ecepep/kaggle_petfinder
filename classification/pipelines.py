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

'''
from classification.transformer import PipeLabelEncoder, DataFrameSelector, InferNA, ToSparse, FnanToStr,\
    AsType, PipeOneHotEncoder, StringConcat, Ravel
from classification.custom_nn_base import CustomNNBase
from classification.custom_nn_categorical import CustomNNCategorical

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import tensorflow as tf
from classification.custom_nn_ordered import CustomNNordered
from sklearn.feature_extraction.text import TfidfVectorizer

import copy
from sklearn.decomposition.truncated_svd import TruncatedSVD

# csv features' type
numeric_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
ordinal_features = ['MaturitySize', 'FurLength', 'Health']

print("separation low high to be questionned for breed and state, I wonder if they ain't the same num of label. Incoherence? @todo")
breed = ['Breed1', 'Breed2']
color = ['Color1', 'Color2', 'Color3']
h_dim_nom = [ 'State', 'RescuerID']
low_dim_nom = breed + color
nominal_features = breed + color + h_dim_nom

binary_features = ['Gender', 'Vaccinated', 'Dewormed', 'Sterilized']
text_features = ["Name", "Description"]

# metadata preprocesssed features :see preprocessed/metadata.py
expected_len = 10
meta_labels = ["label" + str(i) for i in range(0, expected_len)]
meta_label_scores = ["label_score" + str(i) for i in range(0, expected_len)]

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
            ('astype', AsType(astype = "float64")),
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

print("meta_label_simple_concat_pipe pipe, tfidf param set for mail classification @todo")
meta_label_simple_concat_pipe = Pipeline([
            ('sel_label', DataFrameSelector(meta_labels)),
            ('rm_nan', FnanToStr()),#supposed to be useless,
            ('concat_labels', StringConcat()),
            ("ravel", Ravel()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,5),\
strip_accents='ascii', analyzer= "word", stop_words = None, norm = "l1", use_idf = True)),
])

# list of main base pipes
# num_pipe, nom_pipe_label_encode, h_dim_nom_pipe_label_encode, low_dim_nom_pipe_oh, state_oh, des_pipe, des_pipe_svd, meta_label_simple_concat_pipe
##############################################################################
# pipes for random forest
##############################################################################
pipe_rdf = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode', nom_pipe_label_encode)
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
pipe_rdf_oh.named_steps.clf = RandomForestClassifier(n_estimators = 300)

# test impact of state and rescuer as label on precision
pipe_rdf_low_dim_only = copy.deepcopy(pipe_rdf)
pipe_rdf_low_dim_only.named_steps.u_prep.transformer_list[1] = \
    ('nom_pipe_label_encode_low_dim_scale', nom_pipe_label_encode_low_dim_scale)
    
##############################################################################
# pipes for mlp
##############################################################################

epoch =  500 # 1, 500
if epoch == 1: print("epoch == 1; debugging")

pipe_mlp = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode_scale', nom_pipe_label_encode_scale)
    ])),
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))
])

pipe_mlp_low_dim_only = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe),
        ('nom_pipe_label_encode_low_dim_scale', nom_pipe_label_encode_low_dim_scale)
    ])),
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))
])

pipe_mlp_oh = copy.deepcopy(pipe_mlp)
pipe_mlp_oh.named_steps.u_prep.transformer_list[1] = ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh)

pipe_mlp_oh_wider = copy.deepcopy(pipe_mlp_oh)
pipe_mlp_oh_wider.named_steps.clf = ('clf', CustomNNCategorical(hidden = [500, 300, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))

pipe_mlp_oh_des_svd = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh),
        ('des_pipe_svd', des_pipe_svd),
    ])),
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))
])

pipe_mlp_oh_des = copy.deepcopy(pipe_mlp_oh_des_svd)
pipe_mlp_oh_des.named_steps.u_prep.transformer_list[2] = ('des_pipe', des_pipe)

pipe_mlp_oh_des_svd_meta = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh),
        ('des_pipe_svd', des_pipe_svd),
    ])),
    ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch))
])

def toOrderedMlp(pipeWithCat):
    '''
    change NN from categorical to ordered
    :param pipeWithCat: a pipe with a categoricalNN as clf
    '''
    pipeWithCat = copy.deepcopy(pipeWithCat)
    pipeWithCat.named_steps.clf = CustomNNordered(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch)
    return pipeWithCat
pipe_mlp_ordered = toOrderedMlp(pipe_mlp)
pipe_mlp_oh_des_svd_meta_ordered = toOrderedMlp(pipe_mlp_oh_des_svd_meta)


# raise 'test support of scipy sparse by CustomNNCategorical'
# pipe_mlp_des_oh_ordered = Pipeline([
#     ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     ('u_prep', FeatureUnion([
#         ('num_pipe', num_pipe_sparse),
#         ('nom_pipe_oh_sparse', nom_pipe_oh_sparse),
#         ('des_pipe', des_pipe)
#     ])),
#     ('clf', CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
#                                 reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
#                                 epoch = epoch))
# ])