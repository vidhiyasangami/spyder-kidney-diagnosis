# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:56:31 2018

@author: sangami vivekanandan
"""

import pandas as pd
#dataset
dataset = pd.read_csv("kidney_disease.csv")

#data preprosessing
dataset_factorize = dataset.apply(lambda x:pd.factorize(x)[0])

#splitting of data into features and labels
x = dataset_factorize.iloc[:].values
y = dataset_factorize.iloc[:,-1].values

# implementing train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=66)

#applying randomforestalgorithm(classifier)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)

# predictions
rfc_predict = rfc.predict(x_test)

#obtaining accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y[0:132], rfc_predict)