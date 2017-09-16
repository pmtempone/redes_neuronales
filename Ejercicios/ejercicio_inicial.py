#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:05:32 2017

@author: pablotempone
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

#1
adult = pd.read_excel('/Users/pablotempone/Google Drive/Maestria/Redes Neuronales/Adult_Parte1.xlsx')

x = np.array(adult)

#2
x.shape



#3
x[:,:3].min(axis=0)

#4
x[:,:3].max(axis=1)

#5

x_scale = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))

#6
x_aleatorio = np.random.permutation(x_scale)

#7

T = x_scale[:,3]
X = x_scale[:,:3]

#8

rnd_nro = np.random.uniform(size=11029)

x_train = np.column_stack((X,rnd_nro))
x_train = x_train[x_train[:,3]<=0.8,:3]


x_test = np.column_stack((X,rnd_nro))
x_test = x_test[x_test[:,3]>0.8,:3]

T_train = np.column_stack((T,rnd_nro))
T_train = T_train[T_train[:,1]<=0.8,:1]

T_test = np.column_stack((T,rnd_nro))
T_test = T_test[T_test[:,1]>0.8,:1]

#9 

X_traspuesta = X.T

x_train_T = x_train.T

x_test_T = x_test.T
T_train_T = T_train.T
T_test_T = T_test.T

X_traspuesta.min(axis=1)

X_traspuesta.max(axis=0)

#10

len(T_test)

prediction = np.random.randint(2, size=len(T_test))

#11

np.sum(prediction == T_test[:,0]) #1090 coincidencias

#12

confusion_matrix(prediction,T_test[:,0])