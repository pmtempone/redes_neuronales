#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Sep 30 11:40:21 2017

@author: pablotempone

Title: Actividad 4 - Ejercicio 1
"""

import numpy as np
import Ejercicios.Actividad_3.ImagenPuntos as ip
import PIL
from scipy import misc
import func.cpn as cpn
import pandas as pd
from sklearn.metrics import confusion_matrix

figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 1.bmp')
figura2 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 2.bmp')
figura3 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 3.bmp')
figura4 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 4.bmp')
figura5 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 5.bmp')

#Imagen 1

figura1_scale = (figura1[:,:2]-figura1[:,:2].min(axis=0))/(figura1[:,:2].max(axis=0)-figura1[:,:2].min(axis=0))

P = figura1_scale[:,:2].T
T = figura1[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, w_S) = cpn.train(P,T_matriz, T, 5, 0.1, 0.05, 0.01, 100, 100, True)

(salidas, CantPatrones) = T_matriz.shape
ocultas = 5

# %%
# Predict
# %%
#
print(CantPatrones)

# %%
ganadoraPred = np.array([])
w_SPred = np.array([])

distanciasPred = -np.sqrt(np.sum((w_O - (T_matriz[:, 100][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
ganadora = np.argmax(distanciasPred)

np.sum(w_S[:,ganadora]>0.8)



ganadoraPred = np.append(ganadoraPred, np.argmax(distanciasPred)).astype(int)

T3 = T

for p in range(CantPatrones):
    distanciasPred = -np.sqrt(np.sum((w_O - (T_matriz[:, p][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
    ganadora = np.argmax(distanciasPred)

    w_S[:, ganadora] = w_S[:, ganadora] + 0.05 * (T_matriz[:, p] - w_S[:, ganadora])
    T3[p] = ganadora

w_S[:,1.0]
w_S[:,T3[1].astype(int)]
matriz_pred = np.array([]).reshape(0,salidas)
for i in range(CantPatrones):
    matriz_pred = np.vstack([matriz_pred,w_S[:,T3[i].astype(int)]])

pred = pd.DataFrame(matriz_pred)

resultados = pd.DataFrame(data=None,columns=('negro','rojo'))

resultados['negro'] = np.where(pred[0]>=0.8,1,0)
resultados['rojo'] = np.where(pred[1]>=0.8,1,0)

confusion_matrix(T_matriz.T[:,0],resultados['negro'])

def pred_red(T_matriz,T,ocultas,w_O,w_S):
    T3 = T
    (salidas, CantPatrones) = T_matriz.shape
    for p in range(CantPatrones):
        distanciasPred = -np.sqrt(np.sum((w_O - (T_matriz[:, p][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
        ganadora = np.argmax(distanciasPred)

        w_S[:, ganadora] = w_S[:, ganadora] + 0.05 * (T_matriz[:, p] - w_S[:, ganadora])
        T3[p] = ganadora
    matriz_pred = np.array([]).reshape(0, salidas)
    for i in range(CantPatrones):
        matriz_pred = np.vstack([matriz_pred, w_S[:, T3[i].astype(int)]])


# %%

print(distanciasPred)
print(ganadoraPred)
# %%
Clase = np.array([])

for valor in ganadoraPred:
    if valor == 0 or valor == 2:
        Clase = np.append(Clase, 0)
    else:
        Clase = np.append(Clase, 1)

print(ganadoraPred)
print(Clase)

#Imagen 2

figura2_scale = (figura2[:,:2]-figura2[:,:2].min(axis=0))/(figura2[:,:2].max(axis=0)-figura2[:,:2].min(axis=0))

P = figura2_scale[:,:2].T
T = figura2[:,2]

T_matriz = np.concatenate(([T==0], [T==1],[T==2]), axis=0).astype(int)

(w_O, w_S) = cpn.train(P,T_matriz, T, 7, 0.09, 0.03, 0.02, 100, 100, True)

(salidas, CantPatrones) = T_matriz.shape
ocultas = 7

T3 = T

for p in range(CantPatrones):
    distanciasPred = -np.sqrt(np.sum((w_O - (T_matriz[:, p][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
    ganadora = np.argmax(distanciasPred)

    w_S[:, ganadora] = w_S[:, ganadora] + 0.05 * (T_matriz[:, p] - w_S[:, ganadora])
    T3[p] = ganadora

ganadoraPred = np.array([])
w_SPred = np.array([])

distanciasPred = -np.sqrt(np.sum((w_O - (T_matriz[:, 0][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
ganadora = np.argmax(distanciasPred)

w_S[:,ganadora]>0.8

#Imagen 3

np.random.shuffle(figura3)
figura3_scale = (figura3[:,:2]-figura3[:,:2].min(axis=0))/(figura3[:,:2].max(axis=0)-figura3[:,:2].min(axis=0))

P = figura3_scale[:,:2].T
T = figura3[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, w_S) = cpn.train(P,T_matriz, T, 5, 0.09, 0.03, 0.02, 100, 100, True)

#Imagen 4

np.random.shuffle(figura4)
figura4_scale = (figura4[:,:2]-figura4[:,:2].min(axis=0))/(figura4[:,:2].max(axis=0)-figura4[:,:2].min(axis=0))

P = figura4_scale[:,:2].T
T = figura4[:,2]

T_matriz = np.concatenate(([T==0], [T==1],[T==2], [T==3]), axis=0).astype(int)

(w_O, w_S) = cpn.train(P,T_matriz, T, 8, 0.09, 0.03, 0.02, 100, 100, True)

#Imagen 5

np.random.shuffle(figura5)
figura5_scale = (figura5[:,:2]-figura5[:,:2].min(axis=0))/(figura5[:,:2].max(axis=0)-figura5[:,:2].min(axis=0))

P = figura5_scale[:,:2].T
T = figura5[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, w_S) = cpn.train(P,T_matriz, T, 8, 0.09, 0.03, 0.02, 100, 100, True)
