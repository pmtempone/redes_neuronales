# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:18:21 2017

@author: LMRios
"""
# %%

import numpy as np
from openpyxl import load_workbook
import os

print(os.getcwd())
import Ejercicios.Actividad_3.ImagenPuntos as ip
import PIL
from scipy import misc
import func.cpn as cpn
import pandas as pd
from sklearn.metrics import confusion_matrix
import Ejercicios.Actividad_3.bpn as bpn
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf

# %%
# Actividad 4 - Ejercicio 1
ocultas =4
alfa= 0.4
beta=0.3
gamma=0.01
max_ite_O=100
max_ite_S=100
dibujar=True

# %%
def escalado(dataset):
    maximos_columnas = np.max(dataset, axis=0)
    minimos_columnas = np.min(dataset, axis=0)
    dataset_escalado = (dataset - minimos_columnas) / (maximos_columnas - minimos_columnas)
    return dataset_escalado


# %%

figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 1.bmp')
figura2 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 2.bmp')
figura3 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 3.bmp')
figura4 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 4.bmp')
figura5 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 5.bmp')

v_imagen = figura1
print(v_imagen)
print(v_imagen.shape)
# %%
np.random.shuffle(v_imagen)
X = v_imagen[:, 0:2]
X_escalado = escalado(X)
# %%
# valores de salida
T = v_imagen[:, 2]
Torig = np.copy(T)
print(T)

# %%
T_matriz = np.concatenate(([T == 0], [T == 1]), axis=0).astype(int)
print(T_matriz)
print(T_matriz.shape)
# %%
ocultas = 4
alfa = 0.4
beta = 0.3
gamma = 0.01
max_ite_O = 100
max_ite_S = 100
dibujar = True

# %%
(w_O, w_S) = cpn.train(X_escalado.T, T_matriz, T.T, ocultas, alfa, beta, gamma, max_ite_O, max_ite_S, dibujar)
# %%
print("Los pesos de las capas ocultas")
print(w_O)

print("Los pesos de las capas de salida")
print(w_S)

# %%
# Predict
Pred = X_escalado[0:5, :].T
print(Pred)
# %%
(entran, CantPatrones) = Pred.shape
print(entran)
print(CantPatrones)

# %%
ganadoraPred = np.array([])
w_SPred = np.array([])

for p in range(CantPatrones):
    distanciasPred = -np.sqrt(np.sum((w_O - (Pred[:, p][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
    ganadoraPred = np.append(ganadoraPred, np.argmax(distanciasPred)).astype(int)
# %%

print(distanciasPred)
print(ganadoraPred)

# %%
ganadoraPredProb = np.zeros((CantPatrones, entran))
for valor in range(CantPatrones):
    ganadoraPredProb[valor, :] = w_S[:, ganadoraPred[valor]]

print(ganadoraPredProb)
# %%

ganadoraPredProb[ganadoraPredProb >= 0.8] = 1
ganadoraPredProb[ganadoraPredProb <= 0.2] = 0

print(ganadoraPredProb)

# %%
resultado = (ganadoraPredProb == T_matriz[:, 0:5].T)
print(ganadoraPredProb)
print(T_matriz[:, 0:5].T)

print(resultado)

# %%
print("La taza de aciertos del Predict es de :")
np.sum(np.sum(resultado, axis=1) == 2)
# %%

for valor in ganadoraPred:
    ganadoraPredProb = np.append(ganadoraPredProb, w_S[:, valor])
ganadoraPredProb = ganadoraPredProb[np.newaxis]
print(ganadoraPredProb)

# %%
Clase = np.array([])

for valor in ganadoraPred:
    if valor == 0 or valor == 2:
        Clase = np.append(Clase, 0)
    else:
        Clase = np.append(Clase, 1)

print(ganadoraPred)
print(Clase)

# %%
w_SPred[:, ganadoraPred] = w_S[:, ganadoraPred] + beta * (T_matriz[:,p] - w_S[:, ganadoraPred])
#    T3[p] = ganadora
# %%

print(w_SPred)


######ejercicio 1######
#######################

def escalado(dataset):
    maximos_columnas = np.max(dataset, axis=0)
    minimos_columnas = np.min(dataset, axis=0)
    dataset_escalado = (dataset - minimos_columnas) / (maximos_columnas - minimos_columnas)
    return dataset_escalado


# %%

figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 1.bmp')
figura2 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 2.bmp')
figura3 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 3.bmp')
figura4 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 4.bmp')
figura5 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 5.bmp')

v_imagen = figura1
v_imagen = figura2
v_imagen = figura3

print(v_imagen)
print(v_imagen.shape)
# %%
np.random.shuffle(v_imagen)
X = v_imagen[:, 0:2]
X_escalado = escalado(X)
# %%
# valores de salida
T = v_imagen[:, 2]
Torig = np.copy(T)
print(T)
print
# %%
T_matriz = np.concatenate(([T == 0], [T == 1]), axis=0).astype(int)
print(T_matriz)
print(T_matriz.shape)
# %%
ocultas = 5
alfa = 0.4
beta = 0.3
gamma = 0.01
max_ite_O = 100
max_ite_S = 100
dibujar = True

# %%
(w_O, w_S) = cpn.train(X_escalado.T, T_matriz, T.T, ocultas, alfa, beta, gamma, max_ite_O, max_ite_S, dibujar)
print("Los pesos de las capas ocultas")
print(w_O)

print("Los pesos de las capas de salida")
print(w_S)

# %%
# Predict
Pred = X_escalado[0:5, :].T
# Pred= X_escalado[:, :].T

print(Pred)
# %%
(entran, CantPatrones) = Pred.shape
print(entran)
print(CantPatrones)

# %%
ganadoraPred = np.array([])
w_SPred = np.array([])

for p in range(CantPatrones):
    distanciasPred = -np.sqrt(np.sum((w_O - (Pred[:, p][np.newaxis]) * np.ones((ocultas, 1))) ** 2, 1))
    ganadoraPred = np.append(ganadoraPred, np.argmax(distanciasPred)).astype(int)
# %%

print(distanciasPred)
print(ganadoraPred)

# %%
ganadoraPredProb = np.zeros((CantPatrones, entran))
for valor in range(CantPatrones):
    ganadoraPredProb[valor, :] = w_S[:, ganadoraPred[valor]]

print(ganadoraPred)
print(ganadoraPredProb)
# %%

ganadoraPredProb[ganadoraPredProb >= 0.7] = 1
ganadoraPredProb[ganadoraPredProb <= 0.3] = 0

print(ganadoraPredProb)

# %%
resultado = (ganadoraPredProb == T_matriz[:, 0:5].T)
print(ganadoraPredProb)
print(T_matriz[:, 0:5].T)
print(resultado)

# %%
print("La taza de aciertos del Predict es de :")
np.sum(np.sum(resultado, axis=1) == 2)