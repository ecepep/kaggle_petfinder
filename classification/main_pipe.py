'''
Created on Jan 14, 2019

@author: ppc

main_pipe allow to run and assess the different pipelines

@todo consider Amazon aws EC2
@todo lasso for feature selection
@todo meta_only == 0.100, worthless?
@todo take a glance at misclassified pets
@todo rescu pipe / breed, color / state; add high to mlp_oh anyway (pipe_mlp_low_dom_only <0.2)
@todo to reduce patience could make a more complexe value in the logger which also pay attention wheather loss is 
on a plateau
@todo wright a transformer that check for ~~~standard scaling

@todo
- tune dropout at the end
- print grid search implement print all result and gen
- @see pipelines todos
- make a human readable output to save result of previously executed run with param
- ask on stacko comprehension list func scope
- @see bench and tf_test todos
- dir paths' hard coded
- data augment from web petfinder?
     
'''

from classification.util import  fitPrintPipe, getTrainTest2_meta_img_rn, get_from_pkl, \
    check_generalization, printCatGen, quadratic_cohen_kappa

import sys

from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection._split import ShuffleSplit

from multiprocessing import cpu_count
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt

#################################################################"
#definition of all pipelines
from classification.pipelines import *
from time import sleep
#################################################################"

pathToAll = "../all" # path to dataset dir
# train, test = getTrainTest2_meta_img_rn(pathToAll)
train, test = get_from_pkl(pathToAll, v = "getTrainTest2_meta_img_rn")

# multiprocessing requires the fork to happen in a __main__ protected
if __name__ == "__main__":
##############################################################################
# random forest
##############################################################################
    parameters = {
#          'clf__n_estimators': (200, 350), # high influence. 100: kappa ==> 0.340, 200 ==> 0.386, 200>300
# ValueError: n_components must be < n_features; got 140 >= 124
#         'u_prep__des_pipe_svd_v2__SVD__n_components': (25,50,100),
#         'u_prep__des_pipe__tfid_vect__max_df': (0.7, 0.743, 0.775),
#         'clf__min_samples_leaf': (1, 5, 10, 30,) # . 1 lower better
#         "clf__max_features":("sqrt", "log2", 2,3,4,5,6) # auto = sqrt, seems fine
#         "u_prep__pipe_img_PCA__PCA__n_components":(10,20,30,40,50,70,100,200) # pipe_rdf_img_PCA
#          "pipe_img_PCA__PCA__n_components":(10,11),#300, 500, 750, 1000)
    }
    
    DEBUG = False
    if DEBUG:
        n_cpu = 1
        cv_gs = ShuffleSplit(n_splits=1, test_size=0.3, random_state=None) # for testing
        print("DEBUG no multithread, no cv")
    else:
        n_cpu = cpu_count()-1 #  1   cpu_count()-1
        cv_gs = 3 # strat cross-val # necessary
    
    qwk_scorer = make_scorer(quadratic_cohen_kappa)
      
    pipe = {
# #         "pipe_rn_cmp":pipe_rn_cmp, 
#         "pipe_rdf":pipe_rdf, 
#         "pipe_rdf_extra_dim":pipe_rdf_extra_dim, # show extra rn dim influence
#         "pipe_rdf_img":pipe_rdf_img, 
#         "pipe_rdf_img_PCA":pipe_rdf_img_PCA, 
           
#         "pipe_rdf_img_only":pipe_rdf_img_only, 
#         "pipe_rdf_des_only":pipe_rdf_des_only,
#         "pipe_rdf_binary_only":pipe_rdf_binary_only,
#         "pipe_rdf_real_num_only":pipe_rdf_real_num_only,
#         "pipe_rdf_meta_only":pipe_rdf_meta_only,
#         "pipe_rdf_img_PCA_only":pipe_rdf_img_PCA_only,  
 
#         "pipe_rdf_oh":pipe_rdf_oh, 
#         "pipe_rdf_des_svd":pipe_rdf_des_svd, 
#         "pipe_rdf_des_svd_v2": pipe_rdf_des_svd_v2,
#         "pipe_rdf_des_svd_v3": pipe_rdf_des_svd_v3,
#         "pipe_rdf_des_svd_meta":pipe_rdf_des_svd_meta,
#         "pipe_rdf_low_dim_only":pipe_rdf_low_dim_only, 
#         "pipe_rdf_des":pipe_rdf_des,
    }
    for p in pipe.keys():
        print("_ " + p + "______________________________________________")
        fitPrintPipe(pipe[p], X = train, y = train["AdoptionSpeed"], 
                     scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
  
            
##############################################################################
# Custom MLP categorical_crossentropy and ordered with binary crossentropy
#############################################################################    

    # du to very long computation with mlp, run outputs are save to a file
    if DEBUG:
        res_path = None
        n_run = 1
    else:
        res_path = pathToAll + "/result/"
        print("res_path", res_path)
        sys.stdout = open(res_path + "mlp", 'w') # console output to file
        print("WARNING n_run =1")
        n_run = 2 # 3
    
    n_cpu = 1 # mutlithreading already implemented through keras
    parameters = {}

    pipe = {
        "pipe_mlp_oh":pipe_mlp_oh,
#         "pipe_mlp_ohbis":pipe_mlp_oh,
#         "pipe_mlp_ohter":pipe_mlp_oh, 
        "pipe_mlp":pipe_mlp, 
        "pipe_mlp_low_dim_only":pipe_mlp_low_dim_only, 
        "pipe_mlp_oh_des":pipe_mlp_oh_des, 
        "pipe_mlp_oh_des_svd":pipe_mlp_oh_des_svd,
               
#         "pipe_mlp_oh_img":pipe_mlp_oh_img, # too high dim
#         "pipe_mlp_img_only":pipe_mlp_img_only,  # too high dim
              
        "pipe_mlp_oh_img_PCA":pipe_mlp_oh_img_PCA, 
        "pipe_mlp_oh_des_img_PCA":pipe_mlp_oh_des_img_PCA, 
#         "pipe_mlp_doStrong_oh_des_img_PCA":pipe_mlp_doStrong_oh_des_img_PCA, 
        "pipe_mlp_simple_oh_des_img_PCA":pipe_mlp_simpler_oh_des_img_PCA, 
     
#         "pipe_mlp_DoStrongNN_oh":pipe_mlp_DoStrongNN_oh,
#         "pipe_mlp_DoWeakNN_oh":pipe_mlp_DoWeakNN_oh,
        "pipe_mlp_reguNN_oh":pipe_mlp_reguNN_oh,
#         "pipe_mlp_oh_input08NN":pipe_mlp_oh_input08NN,
#         "pipe_mlp_oh_inputnaNN":pipe_mlp_oh_inputnaNN,
#         "pipe_mlp_oh_inputna01NN":pipe_mlp_oh_inputna01NN,
#         "pipe_mlp_oh_lowDoNN":pipe_mlp_oh_lowDoNN,
#         "pipe_mlp_oh_DoButlastNN":pipe_mlp_oh_DoButlastNN,
        
        "pipe_mlp_simple_oh":pipe_mlp_simpler_oh, 
        "pipe_mlp_oh_wider":pipe_mlp_oh_wider, 
        "pipe_mlp_oh_shallowerNN":pipe_mlp_oh_shallowerNN,
        "pipe_mlp_oh_shallower2NN":pipe_mlp_oh_shallower2NN,
        "pipe_mlp_oh_shallowerWiderNN":pipe_mlp_oh_shallowerWiderNN,
        "pipe_mlp_oh_shallowerWider2NN":pipe_mlp_oh_shallowerWider2NN,

        "pipe_mlp_oh_lossOCCNN":pipe_mlp_oh_lossOCCNN,
        "pipe_mlp_oh_lossOCCQuadraticNN":pipe_mlp_oh_lossOCCQuadraticNN,
        
        "pipe_mlp_orderedNN_oh":pipe_mlp_orderedNN_oh,
        "pipe_mlp_dummyesNN_oh":pipe_mlp_dummyesNN_oh    
    }
            
    
    plots = []
    for p in pipe.keys():
        print("_ " + p + "______________________________________________")
#         fitPrintPipe(pipe[p], X = train, y = train["AdoptionSpeed"], 
#                      scoring = qwk_scorer, cv = cv_gs, n_jobs = n_cpu, verbose=1, parameters = parameters)
        for i in range(0, n_run):
            # not a stratified cv
            print("run n", i, "=====")
            dishonnest = False # if dishonnest small leakage @see dishonnest_validation_mlp
            print("dishonest:",dishonnest)
            gen = check_generalization(
                copy.deepcopy(pipe[p]), 
                metric = quadratic_cohen_kappa, 
                X = train.copy(), y = train["AdoptionSpeed"],
                test_size = 0.4 if n_run <3 else 0.3,
                dishonnest_validation_mlp = dishonnest )
    
            # file where to save plots of accs
            saving_file = None if res_path is None else res_path + "/plot/" + p + str(i)
            plots.append(printCatGen(gen, hist = True, plotname = p, saving_file = saving_file))
    
    if len(plots) > 0:
        if DEBUG:
            print("Done")
            sleep(200) # let some times to see the plots
    
    
            
##############################################################################
# check generalization 
##############################################################################
#     print("rdf")

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
# run result and observations/conclusions
############################################################################
"""
observation on the 2 first run for all pipe.

@todo if debug no plot 
@todo need more smoothing more patience @see plot pipe_mlp_orderedNN
@todo pb with learning rate decrease @see plot pipe_mlp_loss_quadraticOCCNN # monitor loss or cohen kappa if increase > reduce lr
@todo still overfit du to too many dim. MLP scores are getting close from rn_forest


input do inputna01NN > inputnaNN > input05NN > input08NN
DoButLast not interesting.
"""

"""

"""

"""
pipe_mlp_oh
run n 0 =====
dishonest: False
after  85 epochs.
gen['score_train']) 0.48673534361641146
gen['score_test']) 0.31256946673950836
run n 1 =====
after  63 epochs.
gen['score_train']) 0.4204818291860364
gen['score_test']) 0.30848160657652357
Control on dishonnesty, self.patience last epochs
[0.3402858395783369, 0.3280302973200908, 0.32919424358961913, 0.32406336440801986, 0.32523906913085276, 0.2941826438089198, 0.2979014340413657, 0.32959656392824377, 0.31320418197190036, 0.30513683206612463, 0.293020401238943, 0.2894573586100435, 0.30684898519010995, 0.3132036310936036, 0.3204270530201513, 0.3097443944835079, 0.31088861161432935, 0.2906356318287686, 0.2997087318115208, 0.3122443709584881, 0.31959839449541283, 0.3232969802132172, 0.32158047358824804, 0.30029220958186287, 0.31171872903344244, 0.31064699612754954, 0.3048962114422199, 0.30086042364587307, 0.3195752214548808, 0.31325241323440467, 0.3174194307550158, 0.29698156823852273, 0.2995515936179267, 0.3060312812451337, 0.3054174558527244, 0.30356778317008637, 0.30349395374041344, 0.29988638318404914, 0.2976866035129867, 0.31253098818879255, 0.3008145223518931]
run n 0 =====
after  69 epochs.
gen['score_train']) 0.43253154070136757
gen['score_test']) 0.31416633210821754
Control on dishonnesty, self.patience last epochs
[0.2870433634047139, 0.2828893982318522, 0.2840402743304923, 0.2593185474073848, 0.2651206365071578, 0.25983259832598327, 0.2741420306348745, 0.24849180200532073, 0.26398495001299604, 0.23209202598110723, 0.24385979873784736, 0.25510227768461413, 0.2614651982150227, 0.24687121197697515, 0.24915038482684948, 0.26343059220396703, 0.2705891587518643, 0.2782184290183537, 0.26216777398261437, 0.2603795433051883, 0.26384856441094273, 0.2577889035682407, 0.27511466211993196, 0.2490754972531236, 0.2668670671852905, 0.2615686630125208, 0.2478367575025764, 0.25271849872888763, 0.266294062447034, 0.265374423347863, 0.264324813860662, 0.24753523926109766, 0.26099712130830743, 0.262899568252431, 0.2666379125141243, 0.2649632086503405, 0.24663081801227482, 0.24777532592539242, 0.2515362497119408, 0.2313031803034714, 0.2629918200493525]
gen['score_train']) 0.48539249325989764
gen['score_test']) 0.30324322327681663
Control on dishonnesty, self.patience last epochs
[0.3357306734417662, 0.3162541763822184, 0.316987131058054, 0.2997028492483561, 0.31782864584754267, 0.28701229801496586, 0.29202755834764926, 0.3209227419032672, 0.2903597277038725, 0.3029450359980328, 0.3191792076148685, 0.31745276220265184, 0.2958247311510217, 0.29698888013270885, 0.30771557641828895, 0.317017684835615, 0.3064707054357503, 0.2900674425414599, 0.318085486866753, 0.3000559396311133, 0.28133638208661105, 0.2857726835438432, 0.314989460354719, 0.3252675724319247, 0.3117760336910711, 0.3060287600214677, 0.3037048230470766, 0.2860672811609368, 0.30671818209984947, 0.31575476734424546, 0.3156237515246867, 0.3136800865747684, 0.3082748113682827, 0.3076985478756329, 0.3292278955565119, 0.3033273215021909, 0.3110345932425157, 0.31505505911107035, 0.3132187928477488, 0.29520668916811854, 0.312521055689413]



mlp_oh dishonnest 
run n 0 =====
~~0.34
run n 1 =====
Actuall num epoch: 58 
gen['score_train']) 0.4237800031035026
gen['score_test']) 0.32214903704912956
Control on dishonnesty, patience last runs
[0.32214903704912956, 0.30512919163166763, 0.3090641499158714, 0.3075696239190634, 0.312130816784541, 0.2975388405920818, 0.30766354194420087, 0.31487913946923585, 0.3180229494837892, 0.311193468426625, 0.3008754772820824, 0.31281030986597813, 0.2999801904010907, 0.3068275715199561, 0.31440012735909517, 0.2960977140124815, 0.29953889597669237, 0.31119075707189114, 0.3083623649599886, 0.2934098034609788, 0.31096342748027705, 0.3103687398583793, 0.29489542382490874, 0.30039945591542405, 0.31207482255601393, 0.30666211073424976, 0.3071021198213485, 0.306388380748866, 0.3041898467787353, 0.30249280768706044, 0.3067545831962314, 0.3026262438254623, 0.2883301919087561, 0.29077788524615367, 0.29890198455982053, 0.29532684176175705, 0.2907263473189763, 0.29046289690390215, 0.30010232827010286, 0.291863514668223, 0.2918185063621178]

"""

"""
Keras avg batches metric's score
     -> hence it was falls we first implementation of kappa_cohen in keras (@see Custom_nn_categorical.cohen_kappa_metric()
    from now on is used @see Cohen_kappa_logger -tested
@todo rerun all mlp pipe... :/
"""

"""
:deprecated
first run with !!!n_run =1 and dishonnest = True!!! but 0.6-0.4
_ pipe_mlp_oh______________________________________________
after  80 epochs.
gen['score_train']) 0.3332587892619525
gen['score_test']) 0.2773916358500459
_ pipe_mlp_ohbis______________________________________________
after  75 epochs.
gen['score_train']) 0.3035652066429645
gen['score_test']) 0.2704843397494161
_ pipe_mlp_ohter______________________________________________
after  101 epochs.
gen['score_train']) 0.3649385953710902
gen['score_test']) 0.29597209711184436
_ pipe_mlp_oh______________________________________________
after  95 epochs.
gen['score_train']) 0.3551010639560609
gen['score_test']) 0.3153639220597545
_ pipe_mlp______________________________________________
after  81 epochs.
gen['score_train']) 0.3176588871283781
gen['score_test']) 0.2692425657183969
_ pipe_mlp_low_dim_only______________________________________________
after  121 epochs.
gen['score_train']) 0.008254418398082608
gen['score_test']) 0.008690138254480506
_ pipe_mlp_oh_des______________________________________________
after  90 epochs.
gen['score_train']) 0.391669925527
gen['score_test']) 0.3086979626288997
_ pipe_mlp_oh_des_svd____________________________________________
after  74 epochs.
gen['score_train']) 0.3751847157447237
gen['score_test']) 0.29265386332795273
_ pipe_mlp_oh_img______________________________________________
after  124 epochs.
gen['score_train']) 0.302697302665517
gen['score_test']) 0.1930845284452235
_ pipe_mlp_img_only______________________________________________
after  88 epochs.
gen['score_train']) 0.12690297822613306
gen['score_test']) 0.11615014776158994
_ pipe_mlp_oh_img_PCA______________________________________________
after  185 epochs.
gen['score_train']) 0.27870926147267494
gen['score_test']) 0.24976480502203469
_ pipe_mlp_oh_des_img_PCA______________________________________________
after  148 epochs.
gen['score_train']) 0.35889868902777666
gen['score_test']) 0.2703771124730131
_ pipe_mlp_simple_oh_des_img_PCA_______________________________________
after  118 epochs.
gen['score_train']) 0.42104632321797086
gen['score_test']) 0.2905219799274449
_ pipe_mlp_simple_oh______________________________________________
after  99 epochs.
gen['score_train']) 0.36362989832210924
gen['score_test']) 0.2789403877051232
_ pipe_mlp_oh_wider______________________________________________
after  72 epochs.
gen['score_train']) 0.34057693051363835
gen['score_test']) 0.2888525314603484
_ pipe_mlp_reguNN_oh______________________________________________
after  41 epochs.
gen['score_train']) 0.0
gen['score_test']) 0.0
_ pipe_mlp_oh_input08NN______________________________________________
after  76 epochs.
gen['score_train']) 0.23351378005807533
gen['score_test']) 0.22835493065825008
_ pipe_mlp_oh_inputnaNN______________________________________________
after  52 epochs.
gen['score_train']) 0.4247910643328835
gen['score_test']) 0.33637075874288025
_ pipe_mlp_oh_inputna01NN______________________________________________
after  61 epochs.
gen['score_train']) 0.44317815341557
gen['score_test']) 0.3504693297917212
_ pipe_mlp_oh_lowDoNN______________________________________________
Actuall num epoch: 44
after  44 epochs.
gen['score_train']) 0.3730553325014221
gen['score_test']) 0.3006447171560557
_ pipe_mlp_oh_DoButlastNN______________________________________________
after  74 epochs.
gen['score_train']) 0.35291349084242973
gen['score_test']) 0.2912602293303995
_ pipe_mlp_orderedNN_oh______________________________________________
Actuall num epoch: 99
after  99 epochs.
gen['score_train']) 0.33486690092874094
gen['score_test']) 0.30906503843690447
_ pipe_mlp_dummyesNN_oh______________________________________________
after  300 epochs.
gen['score_train']) 0.3676299057961079
gen['score_test']) 0.2978790207333185
"""

"""
cv = 3
_ pipe_rdf_des_svd______________________________________________
Best score: 0.341
_ pipe_rdf_des_svd_v2______________________________________________
Best score: 0.339
_ pipe_rdf_des______________________________________________
Best score: 0.368
_ pipe_rdf_des_svd_v3______________________________________________
Best score: 0.307
"""

"""
@see comment above @function replace_step implies a rerun of pipe_rdf_svd cmp
"""

"""
Mistake: The early stopping used to be on 'acc' aka training set, with 'val_acc' it is on a validation set 
to be defined. Moreover was added a more correct cohen kappa metric.

pipe_mlp_oh with the DropoutNN clf thus achieved ~~336 cv = 1 where it use to be 0.294 cv = 3 
before correct early stopping with oldBasicNN

This new correct and more precise early stopping might greatly improved the issues with overfitting 
we had and could explain the earlier bad result of mlp especially in high dim.

@todo rerun all mlp pipe... :/
"""

"""
From here basicNN become oldBasicNN to remove regularization.
"""

""" 
Explanation for pipe_rdf_img_only scores 0.

It is du to regularization and absence of scaling.
All those feats (img) come from a very large network hence it has very low values.
In absence of scaling it remains close to 0 and NN cannot compensate for it because of reg penalisation
because weights would then skyrocket.

@todo 
It also shows the evidence of dropout superiority other regularisation to bring forward 
features with a lot of information (coming from low_dim) against high dim (des, img, high level one hot)
A second possibility (already considered in one of the todos) would be to make a complexe mlp structure
with more important input coming in later layers :(scheme)

high dim -o-mlp with reg-o-\
                            \_o_mlp with do _o_ softmax 
                            /
low dim -------------------/
"""

"""
to put in perspective with explained_variation @see plots.py
1000 0.097
300 0.110
40 0.112
20 0.110
11 0.098
6 -> 0.85
_ pipe_rdf_img_PCA_only______________________________________________
{'pipe_img_PCA__PCA__n_components': (5, 10, 15, 20)}
Best score: 0.110
    pipe_img_PCA__PCA__n_components: 20
_ pipe_rdf_img_PCA_only______________________________________________
{'pipe_img_PCA__PCA__n_components': (300, 500, 750, 1000)}
Best score: 0.110
    pipe_img_PCA__PCA__n_components: 300
_ pipe_rdf_img_PCA_only______________________________________________
{'pipe_img_PCA__PCA__n_components': (20, 30, 50, 100, 200)}
Best score: 0.114
pipe_img_PCA__PCA__n_components: 200

"""

"""
_ pipe_rdf_img_only______________________________________________
Best score: 0.148
_ pipe_rdf_real_num_only______________________________________________
Best score: 0.155
_ pipe_rdf_meta_only______________________________________________
Best score: 0.100
_ pipe_rdf_img_PCA_only______________________________________________
Best score: 0.112
_ pipe_rdf_img_only______________________________________________
Best score: 0.145
_ pipe_rdf_des_only______________________________________________
Best score: 0.195
_ pipe_rdf_binary_only______________________________________________
Best score: 0.190

"""

"""
Testing generalization on mlp showed that we again have strong problem eventhough regularization and dropout (whatever there strength)
 
"""

"""
epoch 15
oh   oh_img
320 280
302 275
281 310
285 268
281 307
"""

"""
Gridsearch on pipe_rdf_img_PCA's PCA n_components
_ pipe_rdf_img_PCA______________________________________________
parameters:
{'u_prep__pipe_img_PCA__PCA__n_components': (10, 20, 30, 40, 50, 70, 100, 200)}
Best score: 0.319
Best parameters set:
    u_prep__pipe_img_PCA__PCA__n_components: 10

"pipe_rdf":340 > "pipe_rdf_img_PCA":319 (10extra dim)
 > "pipe_rdf_img_PCA":299 (50extra dim) ~~ "pipe_rdf_extra_dim":303 (10 extra dim) 

# when plotted explained variance against n_components gives a ~log. 
With n_cpnts = 50 ==> explain ~~80%; 10=>45%; 20=>60%, 30=>70%
"""


"""
Testing pipe for preprocessed img with cv = 3, PCA(n_comp = 50)

"pipe_rdf":335 > "pipe_rdf_img_PCA":299 (50extra dim) ~~ "pipe_rdf_extra_dim":303 (10 extra) 
"pipe_rdf_img":262 (1000 extra) >"pipe_rdf_img_only":152

(for cmp: pipe_rdf_des_only 0.213)

Conclusion:
As expected, img have some info but kills dim
"""


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

Du to variance of result, > can be questionable

Conclusion:
pipe_mlp_oh > pipe_mlp_low_dim_only => for low dim, oh still works better with mlp
pipe_mlp_oh 287 > pipe_mlp_oh_deeper 273 => wider mlp can't train
pipe_mlp_oh_des 294 > pipe_mlp_oh_des_svd 288 => svd still worse 
pipe_mlp_oh_des_svd_meta 295 ~ pipe_mlp_oh_des 294 => meta is maybe slightly intersting or NN lesser affected by a few more dimension 
pipe_mlp >> everything: always dim issues and/or importance of high dim features state and rescuer 
@todo add labelencoded high dim as a features and compare to pipe_mlp
@todo always print epoch for mlp
@todo compare each to rdf 
"""

"""
n_feat:
- tf-ifd for des ~= 128
- tf-idf for meta ~= 9
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
 _ pipe_rdf_des_only______________________________________________
Best score: 0.213
    clf__n_estimators: 200

____________   
conclusion:
pipe_rdf_des 373 > pipe_rdf 352 > pipe_rdf_des_svd 316 > pipe_rdf_low_dim_only 307 > 
pipe_rdf_des_svd_meta 294 > pipe_rdf_oh 291 > pipe_rdf_des_only 213


know high dim?
true: pipe_rdf, pipe_rdf_des_svd, pipe_rdf_des_svd_meta, pipe_rdf_des
false: pipe_rdf_low_dim_only, pipe_rdf_oh pipe_rdf_des_only

pipe_rdf_low_dim_only >  pipe_rdf_oh: for trees o-h lower scores du to dimension issues.
problem of behaviour in higher dimension overall => same info with more dim lower result significatively
pipe_rdf >> pipe_rdf_low_dim_only: high dimension feats help a lot if as label!!!
pipe_rdf_des > pipe_rdf: des very usefull but shouldn't it curse dim? (or "only" ~~124 extra dim?)
pipe_rdf_des > pipe_rdf_des_svd: SVD actually worsening problem => normal with ncomponent = 100 see comment above

"""

"""
testing result of pipe_rdf variance 
other 11 shuffle split 0.7-0.3
scores: 0.343, 0.307, 0.346, 0.362, 0.333, 0.341, 0.345, 0.333, 0.345, 0.326, 0.349
std = 0.013601166697647297 mean = 0.33909090909090905
0.307 0.362

other 11 cv = 3
scores: 0.348, 0.338, 0.346, 0.348, 0.341, 0.336, 0.338, 0.343, 0.344, 0.344, 0.342
std = 0.003846217418419286 mean = 0.34254545454545454
0.338 0.348
==> cv 3 necessary
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