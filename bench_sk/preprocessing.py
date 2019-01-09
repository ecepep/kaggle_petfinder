'''
Created on Jan 4, 2019

@author: ppc

'''


import pandas as pd
import numpy as np

import sklearn.preprocessing as preprocessing
from pandas.core.series import Series


def readBaseCSV(path, shuffle = True, dropText = True,
                  isNa = {"Health" : 0, "MaturitySize" : 0, "FurLength" : 0, "Gender" : 3,
                                "Vaccinated" : 3, "Dewormed" : 3, "Sterilized" : 3}):
    '''
    read csv and set some to na
    :param path:
    :param shuffle:
    :param dropText:
    :param isNa:
    '''
    df = pd.read_csv(path)
    print("shape initial ", df.shape)
    
    # replace "undefined|not sure" values to na
    for i in isNa.keys():
        toNa = (df[i] == isNa[i]).sum()
        if toNa > 0: print("For",i, ",", toNa, "has been set to NA.")
        df[i] = df[i].replace(isNa[i], np.nan)
  
    if shuffle:  df = df.take(np.random.permutation(df.shape[0]))
    if dropText: df = df.drop(["Name", "Description"], axis = 1) # @todo use text.
    return df

# # drop the NaN 
# df  = df .dropna(axis=0, how="any")

def inferNa(train, test, 
            infer = {
                "Health" : "mean", 
                "MaturitySize" : "mean",
                "FurLength" : "mean",
                "Gender" : "mean",
                "Vaccinated" : "mean",
                "Dewormed" : "mean", 
                "Sterilized" : "mean"}):
    '''
    infer na to mean of value (even for unordered value because they are all binary)
    :param train:
    :param test:
    :param infer:
    '''
    for i in infer.keys():
        if infer[i] == "mean":
            data = train[i].append(test[i])
            replacement = data.mean(skipna = True)
        else: raise "todo inferNa"
        train[i] = train[i].replace(np.nan, replacement)
        test[i] = test[i].replace(np.nan, replacement)
    return train, test

def encodeLabel(train, test, nominal_features):
    le = preprocessing.LabelEncoder()
    for col in nominal_features:
        data = train[col].append(test[col])
        le.fit(data)
        train[col] = le.transform(train[col]).astype("object")
        test[col] = le.transform(test[col]).astype("object")
    return train, test

def flattenLabel(train, test, toFlatten, drop = 0):
    '''
    :param df:
    :param nominal_features:
    :param drop: drop column if more than drop unique value to avoid high dimension
    '''
    for i in toFlatten:
        data = train[i].append(test[i])
        if data.nunique() > drop:
            train = train.drop(i, axis = 1)
            test = test.drop(i, axis = 1)
        else: 
            train, test = encodeLabel(train, test, [i]) # @todo dirty code, generify this train, test, data
            data = train[i].append(test[i])
            oe = preprocessing.OneHotEncoder(sparse=False, categories='auto')
            data = data.values.reshape(len(data), 1)
            oe.fit(data)
            def innerFL(oe, df, i):
                dfV = df[i].values.reshape(len(df[i]), 1)                           
                df_encoded = oe.transform(dfV) 
                df_encoded = pd.DataFrame(df_encoded)
                df_encoded.columns = Series(df_encoded.columns).apply((lambda x, i: i+str(x)), i=i)  # force suffixe
                df = df.drop(i, axis = 1)
                df = df.join(df_encoded, rsuffix=i)
                return df
            train = innerFL(oe, train, i)
            test = innerFL(oe, test, i)
    return train, test