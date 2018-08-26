# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:38:44 2018

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
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#applying Neural Network model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=13, kernel_initializer='uniform', activation='relu', input_dim=26))

classifier.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)