'''
Created on Jan 21, 2019

@author: ppc

list of pipelines to be tested in main_pipe


@todo: 
- feature for state (nb inhabitant?)
- add rescuer to description (might be skipped by tf idf)
- add colors and breeds to each other with weights
- include metadata label scores

'''
from classification.transformer import PipeLabelEncoder, DataFrameSelector, InferNA, ToSparse, FnanToStr,\
    AsType, PipeOneHotEncoder, StringConcat
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
breed = ['Breed1', 'Breed2']
color = ['Color1', 'Color2', 'Color3']
h_dim_nom = [ 'State', 'RescuerID']
nominal_features = breed + color + h_dim_nom

binary_features = ['Gender', 'Vaccinated', 'Dewormed', 'Sterilized']
text_features = ["Name", "Description"]

# metadata preprocesssed features :see preprocessed/metadata.py
expected_len = 10
meta_labels = ["label" + str(i) for i in range(0, expected_len)]
meta_label_scores = ["label_score" + str(i) for i in range(0, expected_len)]
feat_metadata = []

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
            ('sel_nom', DataFrameSelector(breed + color)),
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
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,4),\
strip_accents='ascii', analyzer= "word", stop_words='english', norm = "l1", use_idf = True, norm = None))
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
            ('SVD', TruncatedSVD(n_components=100))
        ])

print("meta_label_simple_concat_pipe pipe, tfidf param set for mail classification @todo")
meta_label_simple_concat_pipe = Pipeline([
            ('sel_label', DataFrameSelector(meta_labels, ravel = True)),
            ('rm_nan', FnanToStr()),#supposed to be useless,
            ('concat_labels', StringConcat()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,5),\
strip_accents='ascii', analyzer= "word", stop_words = None, norm = "l1", use_idf = True, norm = None))
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
        ('des_pipe_svd', des_pipe_svd)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])
# add desccription features 
pipe_rdf_des_meta = Pipeline([
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
##############################################################################
# pipes for mlp
##############################################################################

# PipeOneHotEncoder

epoch = 500
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

pipe_mlp_oh = copy.deepcopy(pipe_mlp)
pipe_mlp_oh.named_steps.u_prep.transformer_list[1] = ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh)

pipe_mlp_ordered = copy.deepcopy(pipe_mlp)
pipe_mlp_ordered.named_steps.clf = CustomNNordered(hidden = [400, 200, 100, 50, 20], dropout = [0.3,0.2,0.1], 
                                reg = [0.01,0.005,0.001], h_act = [tf.nn.relu],
                                epoch = epoch)



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