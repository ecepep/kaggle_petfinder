'''
Created on Feb 21, 2019

@author: ppc
'''
##############################################################################
# pipes for random forest
##############################################################################
from classification.pipelines_base import * 
import xgboost as xgb
from xgboost.training import cv


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


# add desccription features 
pipe_rdf_des_meta = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
#     , transformer_weights=None
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ('nom_pipe_label_encode', nom_pipe_label_encode_sparse),
        ('des_pipe', des_pipe),
        ('meta_label_simple_concat_pipe', meta_label_simple_concat_pipe)
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

pipe_rdf_oh_rm_state = copy.deepcopy(pipe_rdf)
pipe_rdf_oh_rm_state.named_steps.u_prep.transformer_list[1] = ("rm_state_oh", rm_state_oh)
pipe_rdf_oh_rm_breed = copy.deepcopy(pipe_rdf)
pipe_rdf_oh_rm_breed.named_steps.u_prep.transformer_list[1] = ("rm_breed_oh", rm_breed_oh)

# test impact of state and rescuer as label on precision
pipe_rdf_low_dim_only = copy.deepcopy(pipe_rdf)
pipe_rdf_low_dim_only.named_steps.u_prep.transformer_list[1] = \
    ('nom_pipe_label_encode_low_dim_scale', nom_pipe_label_encode_low_dim_scale)
    

def to_xgb(pipe):
    '''
#     https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    replace "clf" step in pipe for an xgb
    
    param of xgb were optimized to the pipe_rdf_des thanks to a search.
    
    :param pipe:
    '''

    
    pipe = replace_step(pipe, "clf", ('clf', copy.deepcopy(
         xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic',
                   booster='gbtree', n_jobs=1, nthread=None, gamma=0.1, min_child_weight=5, max_delta_step=0, 
                   subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1, reg_alpha=0.65, reg_lambda=100,
                    scale_pos_weight=1, base_score=0.5, random_state=0, 
                    seed=None, missing=None) 
        )) )
    return pipe
    
    
    

    