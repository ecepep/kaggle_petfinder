'''
Created on Feb 21, 2019

@author: ppc
'''
##############################################################################
# pipes for mlp
##############################################################################

from classification.pipelines_base import *


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

lossOCCNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric", loss = lossOCC,
        metrics = ['cohen_kappa'])
lossOCCQuadraticNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric", loss = lossOCCQuadratic,
        metrics = ['cohen_kappa'])

simplerNN = CustomNNCategorical(hidden = [200, 100, 50, 20], dropout = [0.1,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric",
        metrics = ['cohen_kappa'])

#poor
# shallowerNN = CustomNNCategorical(hidden = [400, 200, 200], dropout = [0.1,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
shallower2NN = CustomNNCategorical(hidden = [200, 200], dropout = [0.1,0.5,0.5], reg = [],
                h_act = [relu], epoch = epoch, cbEarly = "metric",
                metrics = ['cohen_kappa'])
shallower21NN = CustomNNCategorical(hidden = [100, 100], dropout = [0.1,0.5,0.5], reg = [],
                h_act = [relu], epoch = epoch, cbEarly = "metric",
                metrics = ['cohen_kappa'])
shallower3NN = CustomNNCategorical(hidden = [200, 200,100], dropout = [0.1,0.5,0.5,0.5], reg = [],
                h_act = [relu], epoch = epoch, cbEarly = "metric",
                metrics = ['cohen_kappa'])
shallower31NN = CustomNNCategorical(hidden = [100, 100,50], dropout = [0.1,0.5,0.5,0.5], reg = [],
                h_act = [relu], epoch = epoch, cbEarly = "metric",
                metrics = ['cohen_kappa'])
shallower4NN = CustomNNCategorical(hidden = [100,100,100,100,100], dropout = [0.1,0.5,0.5,0.5], reg = [],
                h_act = [relu], epoch = epoch, cbEarly = "metric",
                metrics = ['cohen_kappa'])


# poor
# shallowerWiderNN = CustomNNCategorical(hidden = [400, 400, 400], dropout = [0.1,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# shallowerWider2NN = CustomNNCategorical(hidden = [500, 500], dropout = [0.1,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])

#poor
# widerNN = CustomNNCategorical(hidden = [600, 300, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])



# DoStrongNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.5,0.5,0.5,0.5], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])
# DoWeakNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.1], reg = [],
#                                h_act = [relu], epoch = epoch, cbEarly = "metric",
#                                 metrics = ['cohen_kappa'])

reguNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [],
                               reg = [0.01,0.01,0.01,0.01], h_act = [relu],
                               epoch = epoch, cbEarly = "metric",
                               metrics = ['cohen_kappa'])

orderedNN = CustomNNordered(
        hidden = [400, 200, 100, 50, 20],
        dropout = [0.1,0.5,0.5,0.5,0.5,0.5], reg = [],
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
# pipe_mlp_oh_wider = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(widerNN)) )
# pipe_mlp_oh_shallowerNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerNN)) )
pipe_mlp_oh_shallower2NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower2NN)) )
# pipe_mlp_oh_shallowerWiderNN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerWiderNN)) )
# pipe_mlp_oh_shallowerWider2NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallowerWider2NN)) )
pipe_mlp_oh_shallower3NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower3NN)) )
pipe_mlp_oh_shallower21NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower21NN)) )
pipe_mlp_oh_shallower31NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower31NN)) )
pipe_mlp_oh_shallower4NN = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(shallower4NN)) )


   
pipe_mlp_dummyesNN_oh = replace_step(pipe_mlp_oh, "clf", ('clf', copy.deepcopy(dummyesNN)) )


