
<div style="border: 1px solid black; padding: 10px;">
<h1>Kaggle petfinder adoption prediction</h1>

<h2>Work undertaken</h2>
</div>

<h3>Links</h3>
<a href= "https://github.com/ecepep/kaggle_petfinder">Github's repo</a> <br>
<a href= "https://www.kaggle.com/c/petfinder-adoption-prediction">Kaggle's competition</a> 

<h3>Introduction</h3>

<p>This jupyter summaries my work for the  kaggle competition "petfinder: adoption prediction". Exhaustive uncleaned source code of trials is to be found in the github's repository.</p>
<p>(My work station: 8gb ram, i5 3337u 2*1.8ghz, no gpu)</p>
<p>I am using python 3.6 through conda. The environement will require (keras (>2.2.3), sci-kit, numpy, scipy, tensorflow, xgboost)</p>

<h3>Summary</h3><br>
<a href="#Preprocessing">1) Preprocessing</a><br>
<a href="#Metrics">2) Metrics' definition</a><br>
<a href="#RandomForestClassifier_bench">3) RandomForestClassifier - bench</a><br>
<a href="#CustomNNCategorical">4) CustomNNCategorical</a><br>
<a href="#Other_preprocessed_features ">5) Other preprocessed features </a><br>
<a href="#xgb_features_selection">6) Xgb features selection</a><br>
<a href="#Conclusion">7) Conclusion </a><br>

<h3 id="Preprocessing">Preprocessing</h3>

<ul>
<li>Retrieve data from CSV.</li>
<li>Set to NA unmeaningful values.</li>
<li>Merge the preprocessed metadatas (explanations about metadata preprocessing is a later topic).</li>
<li>Define some transformers to be used in sci-kit Pipeline.</li>
</ul>


```python
def readBaseCSV(path, shuffle = True,
                  isNa = {"Health" : 0, "MaturitySize" : 0, "FurLength" : 0, "Gender" : 3,
                                "Vaccinated" : 3, "Dewormed" : 3, "Sterilized" : 3}):
    df = pd.read_csv(path)
    
    # replace "undefined|not sure" values to na
    for i in isNa.keys():
        toNa = (df[i] == isNa[i]).sum()
        df[i] = df[i].replace(isNa[i], np.nan)
  
    if shuffle:  df = df.take(np.random.permutation(df.shape[0]))
    return df


def get_train_test(path_to_all = "../all/"):
    trainPath = path_to_all + "/train.csv"
    testPath = path_to_all + "/test/test.csv"

    # read csv and set some to na
    train = readBaseCSV(trainPath, shuffle = True)
    test = readBaseCSV(testPath, shuffle = True)
    return train, test

def get_train_test_meta_img(path_to_all = "../all/"):
    meta_dir = path_to_all + "/preprocessed/metadata_label/"
    img_dir = path_to_all + "/preprocessed/transfered_img/"
    
    # read train, test csvs, set unknown to NA and shuffle
    train, test = get_train_test(path_to_all, silent = silent)
    # add features preprocessed from metadata dir :see preprocessed/prep_metadata
    train, test = merge_metadata(meta_dir, train, test)
    # add features from the preprocessed img (through a frozen cnn)
    train, test = merge_img_fcnn(img_dir, train, test)
    
    return train, test
```

<b> transformers: </b> 


```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import re

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select columns in dataframe according to their name. 
    '''
    def __init__(self, attribute_names, dtype=None, ravel = False, regex = False):
        '''
        
        :param attribute_names:
        :param dtype: output type
        :param ravel: convert output shape of (n, 1) to (n,)
        '''
        self.attribute_names = attribute_names
        self.dtype = dtype
        self.ravel = ravel
        self.regex = regex

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.regex:
            X_selected = X.filter(regex=self.attribute_names)
        else:
            X_selected = X[self.attribute_names]
            
        if self.dtype:
            return X_selected.astype(self.dtype).values
        
        if self.ravel & (X_selected.shape[1] == 1):
            return X_selected.values.ravel()
        
        return X_selected.values
    
class StringConcat(BaseEstimator, TransformerMixin):
    '''
    concat several string features to a single string 
    '''
    def __init__(self, sep = " "):
        '''
        :param sep: separator
        '''
        self.sep = sep

    def fit(self, X, y=None):
        return self
    
    def _concat(self, s):
        return self.sep.join(s)
    
    def transform(self, X):
        remove_sep = lambda s: re.sub(r"[^A-Za-z0-9]", "", s)
        X = np.vectorize(remove_sep)(X)
        return np.apply_along_axis(self._concat, axis = 1, arr = X)

class FnanToStr(BaseEstimator, TransformerMixin):
    '''
    replace float nan to ""
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def on_array(x_str):
        '''
        replace float nan to ""
        :param x: 1D array with bool sel
        '''
        # print(list(a[v_is_str(a)])[1] == float("nan")) # false, why??
        is_str = lambda x: not type(x) is str
        v_is_str = np.vectorize(is_str)
        x_str[v_is_str(x_str)] = ""
        return x_str

    def transform(self, X):
#         train.Description.fillna("none")
        if len(X.shape) == 1:        
            return FnanToStr.on_array(X)
        else:
            for i in range(0, X.shape[1]):
                X[:, i] = FnanToStr.on_array(X[:, i])
            return X
        
class Formater(BaseEstimator, TransformerMixin):
    '''
    Base for transformer to change format of X after transform in Pipe
    '''
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X 
    
class Multiplier(Formater):
    def __init__(self, factor):
        self.factor = factor
    def transform(self, X):
        return X * self.factor
    
class DimPrinter(Formater):
    '''
    To be applied just before classification. Print the number of dimension fed to clf.
    '''
    def __init__(self, extra = None):
        self.extra = extra
        self.printOnce = True
    def transform(self, X):
        if self.printOnce:
            if self.extra: print(self.extra)
            print("X dim print", X.shape)
            self.printOnce = False
        return X
    
class ToSparse(Formater):
    def transform(self, X):
        return sparse.csr_matrix(X)
    
class Reshape(Formater):
    def __init__(self, shape = -1):
        '''
        :param shape: -1  to reduce dim of 1
        '''
        self.shape = shape
    def transform(self, X):
        return X.reshape(self.shape)
    
class Ravel(Formater):
    '''
    necessary before tf-idf or AttributeError: 'numpy.ndarray' object has no attribute 'apply'
    '''
    def _ravel(self, X):
        if len(X.shape)==1: return X
        elif X.shape[1] == 1: return X.ravel()
        else: raise "unexpected Ravel()"
    def transform(self, X):
        return self._ravel(X) 
    
class AsType(Formater):
    def __init__(self, astype):
        self.astype = astype
    def transform(self, X):
        return X.astype(self.astype)

class PipeLabelEncoder(BaseEstimator, TransformerMixin):
    """
     Wraps sklearn’s LabelEncoder, but encodes unseen data in your test 
     set as a default factor-level. Accept more than one column.
    
    @note equivalent: class skutil.preprocessed.SafeLabelEncoder
    
    :warning hardcoded
    """
    def __init__(self, silent = True):
        self.values = list()
        super().__init__()
        self.labels = []
        self.silent = silent
        
    def fit(self, Xt, y=None):
        for i in range(0, Xt.shape[1]):
            self.labels.append(np.unique(Xt[:,i]))
        return self
    
    def transform(self, Xt):            
        assert Xt.shape[1] == len(self.labels)
        for i in range(0, Xt.shape[1]):
            # Test set might have values yet unknown to the classifiers
            unknown = np.setdiff1d(np.unique(Xt[:,i]), self.labels[i], assume_unique=True)
            if (len(unknown) > 0) and (not self.silent) : print(len(unknown), "unknown labels found.")
            
            uValues = np.append(self.labels[i], unknown)
            # all unknown values will take the same extra label
            futurLabel = list(range(0, self.labels[i].size)) + [self.labels[i].size]*len(unknown)
            mapping = dict(zip(uValues, futurLabel))
            
            f = lambda i, mapping: mapping[i] 
            Xt[:,i] = np.vectorize(f)(Xt[:,i], mapping)
        return Xt
    
class PipeOneHotEncoder(PipeLabelEncoder):
    """
    Extend PipeLabelEncoder to one hot encoding.
    :warning not using sparse matrix, but np 2D
    :warning hardcoded
    """
    def __init__(self, silent = True):
        PipeLabelEncoder.__init__(self, silent = silent)
    
    def fit(self, Xt, y=None):
        PipeLabelEncoder.fit(self, Xt, y)
        self.nums_label = [len(self.labels[i]) for i in range(0,len(self.labels))] # number of label
        return self
        
    def transform(self, Xt):            
        assert Xt.shape[1] == len(self.labels)
        Xt = PipeLabelEncoder.transform(self, Xt)
        XtOH = np.zeros((Xt.shape[0], sum(self.nums_label)+1)) # Xt in one hot encode notation
        cumsum_len = np.cumsum([0] + self.nums_label[:-1])
        for ji in range(0, Xt.shape[1]):
            for i in range(0, Xt.shape[0]):
                if not Xt[i,ji] > self.nums_label[ji]: # 'unknown' label is still coded as [0,0,0,0,0,0] 
                    XtOH[i, cumsum_len[ji]+Xt[i,ji]] = 1
        
        return XtOH

class ColorBreedOH(BaseEstimator, TransformerMixin):
    '''
    Concat all colors or breeds together in one single one hot encoding 
    to divide their dim by respectively 3 and ~~2
    '''
    def __init__(self, weights, silent = True):
        self.onehot = PipeOneHotEncoder(silent = silent)
        self.weights = weights
                
    def fit(self, Xt, y=None):     
        '''
        learn the level for Breed0, Breed1, Breed2 together
        '''
        self.onehot.fit(np.reshape(Xt, (np.product(Xt.shape), 1)))
        return self
    
    def transform(self, Xt):
        Xt_transform = None
        self.weights = [1]* Xt.shape[1] if self.weights is None else self.weights
        
        for j in range(0, Xt.shape[1]):
            oh_col = self.onehot.transform(np.reshape(Xt[:,j], (Xt[:,j].shape[0], 1)))
            if Xt_transform is None:
                Xt_transform = oh_col * self.weights[j]
            else:
                Xt_transform = Xt_transform + oh_col * self.weights[j]
        
        return Xt_transform
                
class InferNA(BaseEstimator, TransformerMixin):
    '''
    infer na to mean of value (even for unordered value because they are all binary)
    '''
    def __init__(self, attribute_names, method="mean"):
        '''
        :param attribute_names: feature for which NAs will be infered
        :param method:
        '''       
        self.method = method 
        self.attribute_names = attribute_names
        self.replacement =  dict()

    def fit(self, X, y=None):
        assert self.method == "mean"
        for i in self.attribute_names:
            self.replacement[i] = X.loc[:,i].mean(skipna=True) 
        return self

    def transform(self, X, y=None):
        X =  X.copy() 
        for i in self.attribute_names:
            X.loc[:,i] = X.loc[:,i].replace(np.nan, self.replacement[i], inplace = False)
        return X

```

<h3 id="Metrics">Metrics' definition</h3>
<p>The kaggle's competition defined the metric to be a quadratic cohen's kappa. Sci-kit implements it.</p>


```python
import sklearn.metrics as metrics
from sklearn.metrics.scorer import make_scorer

def quadratic_cohen_kappa(y_true, y_pred):
        return metrics.cohen_kappa_score(y_true, y_pred, weights = "quadratic")
    
qwk_scorer = make_scorer(quadratic_cohen_kappa)
```

<h3 id="RandomForestClassifier_bench">RandomForestClassifier - bench</h3>

<b>base for the pipeline</b> 


1) separate features by properties


```python
### csv features
# feat used in num pipe
numeric_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
ordinal_features = ['MaturitySize', 'FurLength', 'Health']
binary_features = ['Gender', 'Vaccinated', 'Dewormed', 'Sterilized']

# unordered nominal features aka factors
breed = ['Breed1', 'Breed2']
color = ['Color1', 'Color2', 'Color3']
low_dim_only = breed + color + ['State']
nominal_features = low_dim_only + ['RescuerID']

# text features (name is never used)
text_features = ["Name", "Description"]

# 
not_a_feat = ["AdoptionSpeed", "PetID"]

# feat set with NAs at get_train_test, NAs to infer as mean of col
feat_with_nas = ordinal_features + binary_features

### preprocessed 
# metadata preprocesssed features
expected_len = 10
meta_labels = ["label" + str(i) for i in range(0, expected_len)]
meta_label_scores = ["label_score" + str(i) for i in range(0, expected_len)]

```

2) define subpipeline which describe further dynamic preprocessing


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Select data that can be considered as numeric. Standardize.
num_pipe = Pipeline([
            ('sel_num', DataFrameSelector(numeric_features + binary_features + ordinal_features, dtype = 'float32')),
            ("scaler", StandardScaler())
        ])

# Label Encode nominal features
nom_pipe_label_encode = Pipeline([
            ('sel_nom', DataFrameSelector(nominal_features)),
            ('encoder', PipeLabelEncoder(silent=True)),
            ('astype', AsType(astype = "float32"))
        ])

# Tf-idf the description
des_pipe = Pipeline([
            ('sel_num', DataFrameSelector(["Description"], ravel = True)),
            ('rm_nan', FnanToStr()),
            ("ravel", Ravel()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,4),\
strip_accents='ascii', analyzer= "word", stop_words='english', norm = "l1", use_idf = True))
        ])

# to sparse array
num_pipe_sparse = Pipeline([
            ('num_pipe', num_pipe),
            ("sparse", ToSparse())
        ])
nom_pipe_label_encode_sparse = Pipeline([
            ('nom_pipe_label_encode', nom_pipe_label_encode),
            ("sparse", ToSparse())
        ])

```

<b>define RandomForestClassifier pipeline.</b>  
<ul>
<li>Infer the value set as NA while preprocessing, to the mean.</li>
<li>Combine tf-idf's description, label encoded nominal and standardized numerical (aka numerical+binary+ordinal) features.</li>
<li>Classify.</li>
</ul>


```python
from sklearn.ensemble.forest import RandomForestClassifier

# add desccription features 
pipe_rdf_des = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ('nom_pipe_label_encode', nom_pipe_label_encode_sparse),
        ('des_pipe', des_pipe)
    ])),
    ('clf', RandomForestClassifier(n_estimators = 200)),
])
```

<b>Run pipeline</b>  


```python
from multiprocessing import cpu_count
from sklearn.model_selection._search import GridSearchCV

path_to_all = "./all"
train, test = get_train_test(path_to_all) # test is for submission only
n_cpu = cpu_count()-1
grid_search = GridSearchCV(pipe_rdf_des, param_grid = {}, scoring = qwk_scorer,
                           cv=3,n_jobs=n_cpu,return_train_score = True) 
grid_search.fit(train, train["AdoptionSpeed"])
print("Best score: %0.3f" % grid_search.best_score_) # ~=0.365

```

    Best score: 0.369


A comparison between the training score (>0.99) and test score (~=0.365) already hint for some issues with overfitting.

<h3 id="CustomNNCategorical">CustomNNCategorical</h3>

<p>@note: With only roughly 15 000 rows, I did not expect a deep neural network to really score better than a bagging tree method but for the sake of learning, I tried to achieve comparable results with both. Indeed, I was looking for some opportunities to experiment the knowledge I acquired with <a href="https://www.coursera.org/specializations/deep-learning">Andrew NG's moocs</a>. </p>

<b>CustomNNCategorical classifier</b>
<p>The class below implements a classifier for sci-kit learn pipeline. Internally, it uses a custom deep neural network made of several Dense layer of keras' library.</p>

This class offers the possibility to:
<ul>
<li>Define the number of layers and their respectiv widths.</li>
<li>Add dropout and/regularization to each layer.</li>
<li>Reduce dynamically the learning rate.</li>
<li>Use cohen kappa as a custom metric. (for early stopping)</li>
<li>Early stop.</li>
<li>"Adam" optimizer is kept as default.</li>
<li>The loss used is "categorical crossentropy". @note: it does not take for account that our output is ordered and metrics is a quadratic kappa.</li>
</ul>



```python
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomNNBase(BaseEstimator, ClassifierMixin):  
    '''
    classdocs
    '''
    def __init__(self, epoch, loss, optimizer, metrics, batch_size):
        '''
        Constructor
        '''
        self.model = None
        
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        
        self.epoch = epoch
        self.batch_size = batch_size
        
    
    def __compile(self, input_shape, output_shape):
        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pass

    def score(self, X, y=None):
#         self.predict(X)
#         res = model.evaluate(x_test, y_test, verbose=0)
        raise "no defined score"
        return None
        
```


```python
from classification.custom_nn_base import CustomNNBase
from NN_loss.ordinal_categorical_crossentropy import lossOCCQuadratic, lossOCC

from sklearn.preprocessing.label import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, InputLayer
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.nn import relu, softmax
import tensorflow as tf

from time import sleep
from keras import regularizers
from sklearn import metrics as metrics
from sklearn.model_selection._split import train_test_split
from classification.Cohen_kappa_logger import Cohen_kappa_logger


class CustomNNCategorical(CustomNNBase):  
    """
    Base for custom sk classifier implementing NN using keras
    
    implement an MLP for classification with custom metric cohen kappa, no custom loss for now @todo
    """

    def __init__(self, hidden=[200, 100, 50, 20], dropout=[0.1, 0.1], reg=[0.05,0.05], h_act=[relu],
                       epoch=500, batch_size=32,  cbEarly="metric", loss="categorical_crossentropy", 
                       optimizer='adam', metrics=['cohen_kappa'], kappa_weights="quadratic",
                       validation = 0.2, smooth_cb = True
                ):
        '''        
        :param hidden:
        :param dropout:  dropout[0] is assigned to input then hidden
        :param reg: ularization 
        :param h_act: hidden_actication
        :param epoch:
        :param batch_size:
        :param cbEarly: "metric" or an EarlyStopping instance
        :param loss:
        :param optimizer:
        :param metrics: "Accuracy" or 'cohen_kappa'
        :param kappa_weights: compatible with sk(ex:"quadratic", None) ignored if metrics != 'cohen_kappa'
        :param smooth_cb: if True EarlyStopping use val_cohen_kappa smoothed (left avg window 3),
         only with val_cohen_kappa
        
        :note restore_best_weights requires keras 2.2.3
        
        '''
        assert loss in ["categorical_crossentropy", lossOCC, lossOCCQuadratic]
        CustomNNBase.__init__(self, epoch, loss, optimizer, metrics, batch_size)
        assert (len(hidden) > 0) & (len(hidden)+1 >= len(dropout)) & \
            (len(hidden) >= len(reg)) & (len(hidden) >= len(h_act))
            
        self.hidden = hidden
        self.dropout = dropout
        self.reg = reg
        self.h_act = h_act
        self.validation = validation
        
        self.final_activation = softmax
        
        self.cbEarly = cbEarly
        self.smooth_cb = smooth_cb
        
        
        self.cbReduceLR = ReduceLROnPlateau(
            monitor='loss', factor=0.8, patience=3,
            verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        self.kappa_weights = kappa_weights
        if len(self.metrics) > 1 : raise "TODO"
    
        
    def __compile(self, input_shape, output_shape):
        ter = lambda x, i: None if len(x) <= i else x[i]
        reg = [regularizers.l2(i) for i in self.reg] #@TODO ALSO USE L1 FOR BETTER FEATURE SELECTION
        h_act = self.h_act * round(len(self.hidden) / len(self.h_act))
        
        self.model = Sequential()
        
        self.model.add(InputLayer(input_shape=(input_shape,)))
        if not ter(self.dropout, 0) is None: self.model.add(Dropout(ter(self.dropout, 0)))
        
        for i in range(0, len(self.hidden)):
            self.model.add(Dense(self.hidden[i], activation=h_act[i],
                                kernel_regularizer=ter(reg, i), bias_regularizer=ter(reg, i)))                       
            if not ter(self.dropout, i+1) is None: self.model.add(Dropout(ter(self.dropout, i+1))) # first for input
        
        self.model.add(Dense(output_shape, activation=self.final_activation))
        
        self.model.compile(optimizer=self.optimizer,
              loss=self.loss,
              metrics=self.metrics)

    def __category_to_output(self, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        target = to_categorical(y, num_classes=np.unique(y).size)
        return target
    
    def __output_to_category(self, output):
        pred = [np.argmax(i) for i in output]
        pred = self.label_encoder.inverse_transform(pred)
        return pred

    def cohen_kappa_metric_keras(self, y_true, y_pred):
        '''
        Do not work as a metric because kappa is not linear and keras make a weighted avg of batches score
        :deprecated @see Cohen_kappa_logger
        '''
        raise "deprecated @see Cohen_kappa_logger"
        return tf.py_func(self.cohen_kappa_score, [y_true, y_pred], tf.float32)       
    
    def cohen_kappa_score(self, y_true, y_pred): 
        raise "deprecated @see Cohen_kappa_logger"
        y_pred = self.__output_to_category(y_pred)
        y_true = self.__output_to_category(y_true)
        
        score = metrics.cohen_kappa_score(y_true, y_pred, weights=self.kappa_weights)
        return score.astype(np.float32)

    def break_on_epoch_n(self, threshold, sec = 60):
        self.n_epoch = len(self.history.history["loss"])
        if self.n_epoch > threshold:
            sleep(sec)  # cool down
    
    def _fit_val(self, X, output):
        # @todo clean below
        if type(self.validation) is float:
            self.history = self.model.fit(
                X, output,
                validation_split = self.validation,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
            
        elif type(self.validation) is tuple:
            assert self.validation[0].shape[1] == X.shape[1], "X_validation must be transformed with prep first"
            self.validation = (self.validation[0], self.__category_to_output(self.validation[1]))
            self.history = self.model.fit(
                X, output,
                validation_data = self.validation,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
            
        elif self.validation is None:
            self.history = self.model.fit(
                X, output,
                epochs=self.epoch,
                batch_size=self.batch_size,
                callbacks=self.callback_list,
                verbose=0)
        else: raise "unknown validation type"
    
    def _kappa_disambiguation(self, X, output):
        '''
        :param X:
        :param output:
        '''
        self.metric_plot = None
        self.patience = 20 #for cbEarly is enoughfrom observation @todo in init
                    
        if self.metrics[0] == "accuracy":
            self.metric_plot = "acc"
            raise "min_delta must be redefined according to val_acc"
            if self.use_smooth_cb:
                raise 'not available for acc self.use_smooth_cb'
            if self.cbEarly == "metric":
                self.cbEarly = EarlyStopping(
                    monitor= 'val_acc' if self.validation else "acc", min_delta=0.0001, 
                    patience=self.patience, verbose=0, mode='auto')
            self.kappa_logger = None
            
        elif self.metrics[0] == 'cohen_kappa':
            self.metrics = None # 'cohen_kappa_metric' cannot be supported @see explication in Cohen_kappa_logger
            self.metric_plot = 'cohen_kappa'
            if self.cbEarly == "metric":
                if self.validation:
                    monitor = "val_cohen_kappa_smoothed" if self.smooth_cb else "val_cohen_kappa" 
                else:
                    if not self.smooth_cb:
                        monitor = "cohen_kappa"
                    else: raise "No cohen_kappa_smoothed"
                print("monitor", monitor)
                self.cbEarly = EarlyStopping(
                                    monitor = monitor if self.validation else "cohen_kappa",
                                    min_delta=0.00000001, patience=self.patience, # a large patience is necessary!
                                    verbose=0, mode='max', restore_best_weights=True)
            
            if type(self.validation) is float:
                X, X_val, output, y_val = train_test_split(X, output, test_size = self.validation)
            elif type(self.validation) is tuple:
                assert self.validation[0].shape[1] == X.shape[1], "X_validation must be transformed with prep first"
                X_val = self.validation[0]
                y_val = self.__category_to_output(self.validation[1])
            elif not self.validation is None: raise "unknown validation type"
            
#             self.validation = None # can slightly reduce computation but need val_loss for callback LRReduceOnPlateau 

            self.kappa_logger = Cohen_kappa_logger(
                 output_to_category=self.__output_to_category,
                 X_train = X, y_train = output, 
                 X_val = X_val, y_val = y_val, 
                 kappa_weights = self.kappa_weights)
            
        else: 
            print(self.metrics[0])
            raise "not implemented"
        return X, output
            
    def fit(self, X, y=None):
        '''
        :param X:
        :param y:
        :param cbEarly: Parameter for early stopping
        '''
        output = self.__category_to_output(y)
        
        X, output = self._kappa_disambiguation(X, output)
        
        output_shape = output.shape[1]
        input_shape = X.shape[1]
        self.__compile(input_shape, output_shape)
      
        self.callback_list = []
        for cb in [self.kappa_logger, self.cbReduceLR, self.cbEarly]:
            if cb: self.callback_list.append(cb)
        
        self._fit_val(X, output)
        
        self.break_on_epoch_n(50)          
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "history")
        except AttributeError:
            raise RuntimeError("Call fit first.")
        
        preds = self.model.predict(X)
        preds = self.__output_to_category(preds)
        return preds

    def plot_history(self, plotname="NN", saving_file=None):
        '''
        :param plotname:
        :param saving_file: filename where to save plots
        :return plt , to avoid carbage collection and closing of the windows
        '''
        history = self.history
        plot = (saving_file is None)
#         print("History acc", history.history['acc'])
#         print("History loss", history.history['loss'])
#         print("History lr", history.history['lr'])
#         print("Acc train (last)", history.history['acc'][-5:-1])

        import matplotlib.pyplot as plt
        
        if plot: plt.ion()
        if plot: plt.show()

        fig = plt.figure()
        plt.grid(True)
        plt.title(plotname)
#         print("possible plot", history.history.keys())
        if self.metric_plot in history.history.keys():
            plt.subplot(221)
            plt.plot(history.history[self.metric_plot])
            plt.ylabel(self.metric_plot + "  ")
            if plot: plt.draw()
        
        if "val_" + self.metric_plot in history.history.keys():
            plt.subplot(222)
    #         print("possible plot", history.history.keys())
            plt.plot(history.history["val_" + self.metric_plot])
            plt.ylabel("val_" + self.metric_plot + "  ")
            if plot: plt.draw()
    
            if False: 
                print("self.patience last epochs")
                print(history.history["val_" + self.metric_plot][-(self.patience+1):])            
        
        plt.subplot(223)
        plt.plot(history.history['loss'])
        plt.ylabel('"loss" ' + "  " + plotname)
        if plot: plt.draw()
        
        plt.subplot(224)
        if "val_cohen_kappa_smoothed" in history.history.keys():
            plt.plot(history.history['val_cohen_kappa_smoothed'])
            plt.ylabel("val_cohen_kappa_smoothed")
        else:
            plt.plot(history.history['lr'])
            plt.ylabel('"lr"' + "  " + plotname)
        if plot: plt.draw()
        if plot: plt.pause(1)

        if saving_file:
            fig.savefig(saving_file)
            plt = None # send to carbage
                      
        return plt
       
    

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-18-f22c4f908adc> in <module>()
          1 from classification.custom_nn_base import CustomNNBase
    ----> 2 from NN_loss.ordinal_categorical_crossentropy import lossOCCQuadratic, lossOCC
          3 
          4 from sklearn.preprocessing.label import LabelEncoder
          5 


    ~/Documents/eclipse-workspace/kaggle_petfinder/NN_loss/ordinal_categorical_crossentropy.py in <module>()
         10 '''
         11 
    ---> 12 from keras import backend
         13 from keras import losses
         14 


    ModuleNotFoundError: No module named 'keras'


<b>CohenKappaLoger</b>
<p>This neural network tend to overfit as well even with dropout and/or regularization. To help with this issue and to reduce computation time(#no gpu), the class CohenKappaLoger implements a custom (quadratic) cohen kappa metric to be used along side keras EarlyStopping. </p><p>First was considered a real custom metrics rather than a callback (@see @deprecated cohen_kappa_metric_keras). However, Keras calculate the metrics by averaging batches' value. Unfortunately, this works with accuracy but not with cohen's kappa which is a non-linear function.</p>


```python
from sklearn.metrics.classification import cohen_kappa_score as sk_cohen_kappa_score
from keras.callbacks import Callback
from numpy import mean

class Cohen_kappa_logger(Callback):
    '''
    Add to the logs "val_cohen_kappa" and "cohen_kappa" at each epoch's end to record cohen's kappa score.
    Works fine along with EarlyStopping. val_cohen_kappa is avg smoothed
    '''
    def __init__(self, output_to_category=None,
                 X_train = None, y_train = None, 
                 X_val = None, y_val = None, 
                 kappa_weights = "quadratic",
                 smooth_window = 5):
        '''
        
        :param output_to_category:
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :param kappa_weights:
        :param smooth_window: help avoiding overfitting with earlier early stopping
        '''
        self.output_to_category = output_to_category or (lambda x: x)
        self.X_val = X_val
        self.X_train = X_train
        self.y_val = y_val
        self.y_train = y_train
        self.kappa_weights = kappa_weights
        self.smooth_window = smooth_window
        if X_val is None: raise 'implement none X_val COhen kappa logger'
        

    def on_epoch_end(self, epoch, logs={}):
        pred_train = self.model.predict(self.X_train)
        pred_val = self.model.predict(self.X_val)
        
        score_train = self.cohen_kappa(pred_train, self.y_train)
        score_val = self.cohen_kappa(pred_val, self.y_val)
        
        if not "val_cohen_kappa" in self.model.history.history.keys(): # not defined on first run
            smoothed_val_score = score_val # mean(self.model.history.history["val_cohen_kappa"][-(self.smooth_window+1):])
        else: 
            last_score = self.model.history.history["val_cohen_kappa"][-(self.smooth_window-1):]
            last_score.append(score_val)
            smoothed_val_score = mean(last_score)
            
        logs["cohen_kappa"]= np.float64(score_train)
        logs["val_cohen_kappa"]= np.float64(score_val)
        logs["val_cohen_kappa_smoothed"]= np.float64(smoothed_val_score)
        return

    def cohen_kappa(self, y_true, y_pred): 
        y_pred = self.output_to_category(y_pred)
        y_true = self.output_to_category(y_true)
           
        return sk_cohen_kappa_score(y_true, y_pred, weights=self.kappa_weights)

```

<b>CustomNNOrdered</b>
<p>This class tries to give some importance to the ordering of the ordinal predicted value by using another representation, loss function and activation. It performs similar scores as the original network. </p>
<i>A more sophisticated method might be more useful here (Weighted kappa loss function for multi-class classification of ordinal data in deep learning, Jordi de la Torrea, Domenec Puiga, Aida Valls || A simple squared-error reformulation for ordinal classification, Christopher Beckham & Christopher Pal).</i>


```python
from classification.custom_nn_categorical import CustomNNCategorical

from tensorflow.nn import sigmoid
import numpy as np
import warnings

class CustomNNordered(CustomNNCategorical):
    '''
    implement an mlp classifier for ordered categorical
    Ys represented as:
        0: [0, 0, 0, 0, 1],1: [0, 0, 0, 1, 1],2: [0, 0, 1, 1, 1]...
    loss: "binary_crossentropy"
    final activation: sigmoid
        
    simplistic solution to infer importance of ordering to the NN.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        self.loss = None
#         assert loss
#         assert cbEarly (if None ==> define else warning acc)
        super(CustomNNordered, self).__init__(*args, **kwargs)

        if not self.loss is None: warnings.warn("loss will be set as 'binary_crossentropy'")
        self.loss = "binary_crossentropy"

        # acc is bad, cb early with lower delta
        if "cbEarly" in kwargs:
            warnings.warn("Accuracy from keras is wrong for CustomNNordered. Choose the monitor wisely")
            
        self.final_activation = sigmoid
    
    def __category_to_output(self, y):
        '''
        :param y: array of target ordered categories. Categories must be number from 0 to n-1 (ordered)
        '''
        n_cat = y.unique().size
        assert set(y.unique()) == set(range(0,n_cat)), \
            "rewrite more exhaustive fun (toOrderedCategorical)"     
        target = [([0]*(c-i) + [1]*i) for i, c in zip(y, [n_cat]*len(y))]
        return np.array(target)
    
    def __output_to_category(self, output):
        pred = output.round().astype(int)
        pred =  [i.sum() for i in pred]
        return pred
    
    def plot_history(self, plotname="NN", saving_file=None):
        warnings.warn("Accuracy from keras is wrong for CustomNNordered. Choose the monitor wisely")
        CustomNNCategorical.plot_history(self, plotname, saving_file)
```

<b>pipeline for mlp</b><br>
<p>I am using one hot encoding with the labeled data. Level for colors and breeds is high. To reduce the number of dimension, the o-h transformers for those features encode all colors and breeds on the same vector (almost no info loss, dim / 2 or 3).</p>



```python
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
state_oh = Pipeline([
            ('sel_nom', DataFrameSelector(["State"])),
            ('encoder', PipeOneHotEncoder(silent=True)),
            ('astype', AsType(astype = "float64")),
        ])
# Rescuer_ID has to many level to be relevant in one hot encoding. Is ignored.
low_dim_nom_pipe_oh= Pipeline([
            ("ulowdim", FeatureUnion([
                ("colors", colors_oh),
                ("breed", breeds_oh),
                ("state", state_oh)
                ]))
        ])

# NN classifier
basicNN = CustomNNCategorical(hidden = [400, 200, 100, 50, 20], dropout = [0.1,0.5,0.5,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric",
        metrics = ['cohen_kappa'])

# NN classifier @see CustomNNordered
orderedNN = CustomNNordered(
        hidden = [400, 200, 100, 50, 20],
        dropout = [0.1,0.5,0.5,0.5,0.5,0.5], reg = [],
        h_act = [relu], epoch = epoch, cbEarly = "metric",
        metrics = ['cohen_kappa'])

# pipe for mlp, similar to pipe_rdf_des but with NN classifier and one hot encoded label
pipe_mlp_oh_des = Pipeline([
    ('infer_na_mean', InferNA(feat_with_nas, method = "mean")), 
    ('u_prep', FeatureUnion([
        ('num_pipe', num_pipe_sparse),
        ("low_dim_nom_pipe_oh", low_dim_nom_pipe_oh),
        ('des_pipe', des_pipe),
    ])),
    ('clf', copy.deepcopy(basicNN))
])

def replace_step(pipe, step_name, new_step):
    '''
    :param pipe: original pipe
    :param step_name: as string :ex "clf"
    :param new_step: the usual tupple :ex ("clf", RandomForest())
    :return a new pipe where the step as been replace
    
    :warning has to used pipe.steps and not pipe.named_steps
    '''
    for step in range(0, len(pipe.steps)):
        if pipe.steps[step][0] ==  step_name:
            new_pipe = copy.deepcopy(pipe)
            new_pipe.steps[step] = new_step
            return  new_pipe
    raise Exception("step not found:"+step_name)

# pipe_mlp_oh_des with alternate NN@see CustomNNordered
pipe_mlp_oh_des_orderedNN = replace_step(pipe_mlp_oh_des, "clf", ('clf', copy.deepcopy(orderedNN)))

```

<b>Observations</b>
<ul>
<li>
basicNN is the most relevant parameterization I found for this problem. The lack of computation's power did not allowed me to precisely optimize the parameterization.
</li>
<li>
Dropout shows itself more efficient and general than the regularization which tend to bring issues in the propagation.
</li>
<li>
A dropout of 0.5 has been chosen since it theoretically optimizes regularization but for the first layer (0.1).
</li>
<li>
The precision of EarlyStopping is really important. A smoothing has been added to the evaluation of the model to garanty an early enough stopping.
</li>
<li>
Empirically, there are some clear evidence that the dimension of the input has a strong impact on the score for both NN and rdf (higher dim -> ovrefit).<br>
The base pipeline (label encoded) has 19+125 features. 125 corresponds to the features from the tf-idfed description. However, the description tend to increase the score by ~0.02/0.03 regardless of the model.
Though the concatenation of breeds and colors in one hot encoding reduce the dimension from more than 100, it curiously performs similar with NN (a dummy test with rdf shows its correct implementation).
</li>
</ul>

<h3 id="Other_preprocessed_features">Other preprocessed features</h3>
<p>In addition to the initial dataset, petfinder provides json's relative to the pet pictures generated from google API. The above mentionned pictures are available as well.</p>
<p>Two possibilities to make of it extra features have been implemented.</p>

<b>google API's Json</b>
<p>From the jsons is extracted the "labelAnnotations" and their respective scores. Those values are saved as pkl for later merging in the whole dataset at the preprocessing step. @see @file /preprocessed/metadata.py</p>
<p>For simplicity and to avoid the curse of dimension. Each picture labels are concatenated to a string which is then processed through TfidfVectorizer.</p>
<p>A more sophisticated preprocessing notably involving the scores could probably yield better results.</p>


```python
class StringConcat(BaseEstimator, TransformerMixin):
    '''
    concat several string features to a single string 
    '''
    def __init__(self, sep = " "):
        '''
        :param sep: separator
        '''
        self.sep = sep

    def fit(self, X, y=None):
        return self
    
    def _concat(self, s):
        return self.sep.join(s)
    
    def transform(self, X):
        remove_sep = lambda s: re.sub(r"[^A-Za-z0-9]", "", s)
        X = np.vectorize(remove_sep)(X)
        return np.apply_along_axis(self._concat, axis = 1, arr = X)

```


```python
meta_label_simple_concat_pipe = Pipeline([
            ('sel_label', DataFrameSelector(meta_labels)),
            ('rm_nan', FnanToStr()),
            ('concat_labels', StringConcat()),
            ("ravel", Ravel()),
            ('tfid_vect', TfidfVectorizer(max_df= 0.743, min_df=0.036, ngram_range=(1,5),\
strip_accents='ascii', analyzer= "word", stop_words = None, norm = "l1", use_idf = True)),
])

pipe_rdf_meta_only = pipe_rdf_extra_dim = Pipeline([
    ('meta_label_simple_concat_pipe', meta_label_simple_concat_pipe),
    ('clf', RandomForestClassifier(n_estimators = 200)),
]) 
```

<i> Observation </i>
<ul>
<li>
<p>The labels' annotations processed by google API are probably unrelevant. It mostly retrieve labels as "dogs", "dogs breed", "carnivoran" (hence the tf-idf) or the name of the breed which is a repetition to an information that already exist in the more reliable  dataset.</p>
</li>
<li>
Though pipe_rdf_meta_only gives a non null prediction (~~0.1), combined with the original pipeline, it performs the same.
</li>

</ul>

<b>pet pictures</b>
<p>@note: Again, this has been implemented for the fun (of using keras) since it is very unlikely to bring valuable result.</p><p> It is aimed at bringing a (slightly) deeper understanding of the pic than the metadata from googles API.</p>
<p>It works as such:</p>
<ul>
<li>For each pet a single picture is retrieved and resized. @see @file /preprocessed/image_transfer.py</li>
<li>A frozen pretrained (on "imagenet") VGG16 prived of its last layer predicts the pic.</li>
<li>Predictions are saved to .pkl to be merged later at preprocessing.</li>
<li>A pca is applied as a desperate solution to avoid adding 1000 features.</li>
</ul>


```python
from keras.applications import VGG16
from keras.layers import Dropout
from keras import optimizers
from keras.models import Sequential
from keras import backend

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize
import sys
from time import sleep
import numpy as np

class FrozenCnn():
    '''
    classdocs
    Predict img from a frozen pretrained cnn 
    '''
    def __init__(self, cnn = VGG16, inputshape = (224,224), n_pop=1):
        '''
        Constructor
        
        :param cnn: cnn to be loaded with imagenet
        :param inputshape: input shape of the CNN, img will be resized to it 
        :param n_pop: number of popped layer
        '''
        self.cnn = cnn
        self.inputshape = inputshape
        if self.cnn == VGG16 and self.inputshape != (224,224): raise "input shape does not match keras VGG16"
        self.n_pop = n_pop
        
    def __compile(self, input_shape, output_shape):
        pass
 
    def fit(self, X = None, y=None):
        if not X is None: print("frozen does not fit to X! (pretrained on imagenet)")
        
        vgg_conv = self.cnn(weights ='imagenet', include_top=True)
        
        for layer in vgg_conv.layers:
            layer.trainable = False
#             print(layer)
#             print("layer out", layer.output)
        
        model = Sequential()
        model.add(vgg_conv)
        
        for i in range(0, self.n_pop):
            model.layers.pop() # Get rid of the classification layer  
            while type(model.layers[-1]) is Dropout:
                model.layers.pop() # Get rid of the dropout layer
        
        print("output:", model.layers[-1].output)
            
        self.model = model   
        return self
 
    def predict(self, X, y=None):
        '''
        use plt.show to show pictures where resize deform with h or w twice bigger
        :param X:
        :param y:
        '''
        # load pretrained cnn from keras
        
        image_resized = []
        for img in X:
            image_resized.append(imresize(img, self.inputshape))
            if img.shape[0] / img.shape[1] > 2 or img.shape[0] / img.shape[1] < 0.5: 
                print("warning on im resize shape,", img.shape)
#                 fig=plt.figure()
#                 fig.add_subplot(1,2,1)
#                 plt.imshow(img)
#                 fig.add_subplot(2,2,2)
#                 plt.imshow(image_resized[-1])
#                 plt.draw()

        prediction = self.model.predict(np.array(image_resized))
#         backend.clear_session() # (was) necessary to avoid memory leakage
        return prediction
           
 
    def score(self, X, y=None):
        pass
    
    def __del__(self):
        backend.clear_session() # could create trouble du to carbage collector being late???
         

```


```python
# pipeline with img's 1000 features, scaled
pipe_img = Pipeline([
    ("sel_imgf", DataFrameSelector("imgf_[0-9]+", regex = True)), 
    ("scaler", StandardScaler())
    ])

# with pca
pipe_img_PCA = Pipeline([("sel_imgf", pipe_img),
                         ("PCA", PCA(n_components=30))])


# with NN classifier
pipe_mlp_img_only = pipe_rdf_extra_dim = Pipeline([
    ("pipe_img", pipe_img),
    ('clf', copy.deepcopy(basicNN)),
]) 
```

<i> Observation </i>
<ul>
<li>
As expected, added to the feature union pipe_img_PCA does not bring any improvements (rather the opposite). However, I was satisfied to see that pipe_mlp_img_only manages better predictions than pipe_rdf_meta_only confirming that predicted images have more info than google's api metadata.
</li>
</ul>

<h3 id="xgb_features_selection">Xgb features selection</h3>
<p>As already mentionned dimensionnality is a first concern here. As a more concrete attempt to improve the score, gradient boosted trees can be used to improve features' selection. For itself, it implements some efficient regularizations and it can even make bet on "features importance". </p>


```python
path_to_all = "./all" # path to dataset dir

train, test = get_train_test_meta_img(path_to_all)
x_train, x_test, y_train, y_test = train_test_split(train, train["AdoptionSpeed"], test_size = 0.3) 

pipe = pipe_rdf_des      

#     https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#    param of xgb were optimized to the pipe_rdf_des thanks to a search.
pipe = replace_step(pipe, "clf", ('clf', copy.deepcopy(
         xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic',
                   booster='gbtree', n_jobs=1, nthread=None, gamma=0.1, min_child_weight=5, max_delta_step=0, 
                   subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1, reg_alpha=0.65, reg_lambda=100,
                    scale_pos_weight=1, base_score=0.5, random_state=0, 
                    seed=None, missing=None) 
        )) )
xgb_clf = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic',
                   booster='gbtree', n_jobs=1, nthread=None, gamma=0.1, min_child_weight=5, max_delta_step=0, 
                   subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1, reg_alpha=0.65, reg_lambda=100,
                    scale_pos_weight=1, base_score=0.5, random_state=0, 
                    seed=None, missing=None) # return a pipeline with the same xgb as in pipe and without prep

pipe.fit(x_train, y_train)    
print("pipe.named_steps.clf.feature_importances_",pipe.named_steps.clf.feature_importances_)

prep_pipe = deepcopy(pipe)
prep_pipe.steps = prep_pipe.steps[:-1]
x_train_tf = prep_pipe.transform(x_train)
x_test_tf = prep_pipe.transform(x_test)    

# X_train_transf = pipe.transform(x_train)
# X_test_transf = pipe.transform(x_test)
threshold = [0, 0.0005, 0.001,0.002, 0.003]
for th in threshold:
    selection = SelectFromModel(pipe.named_steps.clf, threshold=th, prefit=True)
    select_X_train = selection.transform(x_train_tf)
    select_X_test = selection.transform(x_test_tf)
    selection_model = xgb_clf
    selection_model2 = RandomForestClassifier(n_estimators = 200)
    selection_model.fit(select_X_train, y_train)
    selection_model2.fit(select_X_train, y_train)
    
    select_pred_train = selection_model.predict(select_X_train)                                                
    select_pred_test = selection_model.predict(select_X_test)      
    select_pred_train2 = selection_model2.predict(select_X_train)                                                
    select_pred_test2 = selection_model2.predict(select_X_test)                                                  
                                                                         
    select_score_train = quadratic_cohen_kappa(select_pred_train, y_train)                                         
    select_score_test = quadratic_cohen_kappa(select_pred_test, y_test)  
    select_score_train2 = quadratic_cohen_kappa(select_pred_train2, y_train)                                         
    select_score_test2 = quadratic_cohen_kappa(select_pred_test2, y_test)  
    
    print("for threshold", th)
    print("select_score_train", select_score_train)
    print("select_score_test", select_score_test) 
    print("select_score_train rdf", select_score_train2)
    print("select_score_test rdf", select_score_test2) 
```

<b>Observations</b>
<ul>
<li>The features selection do not yield better results.</li>
<li>Xgb performs as well as a rdf classifier.
However, I believe that more precise parameterization could do slightly better.</li>
</ul>

<h3 id="Conclusion">Conclusion</h3>
<p>I took a lot of fun in all the attempts undertaken and it gave me the possibility to experiment with keras.</p>
<p>A more realistic approach of the task would have resulted in higher score but I do not regret the things I learned since they were my true goal.</p>
<p>Du to a lack of time, I cannot bring this project further. However, the todo list in the code is still very long. Some topics as preprocessing, regularization would disserve closer attention and some important steps were overlooked like looking at misclassified predictions or visualizing the fitted model...etc</p>
