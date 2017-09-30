# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: auvimo

Función train
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar la red neuronal. Los ejemplos deben estar en columnas.
       T: es una matriz con la salida esperada para cada ejemplo. Esta matriz 
           debe tener tantas filas como neuronas de salida tenga la red
       T2: clases con su valor original (0 .. n-1) (Solo es utilizado para graficar)
       ocultas: la cantidad de neuronas ocultas que tendrá la red    
       alfa: velocidad de aprendizaje
       momento: término de momento
       fun_oculta: función de activación en las neuronas de la capa oculta
       fun_salida: función de activación en las neuronas de la capa de salida
       MAX_ITERA: la cantidad máxima de iteraciones en las cuales se va a
           ejecutar el algoritmo
       cota_error: error mínimo aceptado para finalizar con el algoritmo
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y las rectas discriminantes.

Devuelve:
       w_O: la matriz de pesos de las neuronas de la capa oculta
       b_O: vector de bias de las neuronas de la capa oculta
       w_S: la matriz de pesos de las neuronas de la capa de salida
       b_S: vector de bias de las neuronas de la capa de salida
       ite: número de iteraciones ejecutadas durante el algoritmo
       errorProm: errorPromedio finalizado el algoritmo

Ejemplo de uso:
       (w_O, b_O, w_S, b_S, ite, errorProm) = train(P, T, 10, 0.25, 1.2, 'logsig', 'tansig', 25000, 0.001, True);
       
"""


import numpy as np
import matplotlib.pyplot as plt

marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm')}

def plot(P, T, W, b, title):
    plt.clf()
    
    #Ejemplos
    for class_value in np.unique(T):
        x = []
        y = []
        for i in range(len(T)):
            if T[i] == class_value:
                x.append(P[0, i])
                y.append(P[1, i])
        plt.scatter(x, y, marker=marcadores[class_value][0], color=marcadores[class_value][1])
    
    #ejes
    minimos = np.min(P, axis=1)
    maximos = np.max(P, axis=1)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #rectas discriminantes
    x1 = minimos[0]
    x2 = maximos[0]
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        m = W[neu,0] / W[neu,1] * -1
        n = b[neu] / W[neu,1] * -1
        y1 = x1 * m + n
        y2 = x2 * m + n
        plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001) 
    
def purelin(x):
    return x

def dpurelin(x):
    return np.ones_like(x)
    
def logsig(x):
    return 1 / (1 + np.exp(-x))

def dlogsig(x):
    return logsig(x) * (1 - logsig(x))

def tansig(x):
    return np.tanh(x)

def dtansig(x):
    return 1.0 - np.square(tansig(x))
    
def train(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, MAX_ITE, cota_error, dibujar):
    (entran, CantPatrones) = P.shape   
    (salidas, CantPatrones) = T.shape   
    
    w_O = np.random.rand(ocultas, entran) - 0.5
    b_O = np.random.rand(ocultas,1) - 0.5
    w_S = np.random.rand(salidas, ocultas) - 0.5
    b_S = np.random.rand(salidas,1) - 0.5
    
    momento_w_S = np.zeros(w_S.shape)
    momento_b_S = np.zeros(b_S.shape)
    momento_w_O = np.zeros(w_O.shape)
    momento_b_O = np.zeros(b_O.shape)
    
    ite = 0;
    ErrorProm = cota_error + 1
    
    while (ite < MAX_ITE) and (ErrorProm > cota_error):
        suma_error = 0
        for p in range(CantPatrones): 
            neta_oculta = w_O.dot(P[:,p][np.newaxis].T) + b_O
            salida_oculta = eval(fun_oculta + '(neta_oculta)')
            neta_salida = w_S.dot(salida_oculta) + b_S
            salida_salida = eval(fun_salida + '(neta_salida)')
           
            error_ejemplo = T[:,p] - salida_salida.T[0]
            suma_error = suma_error + np.sum(error_ejemplo**2)

            delta_salida = error_ejemplo[np.newaxis].T * eval('d' + fun_salida + '(neta_salida)')
            delta_oculta = eval('d' + fun_oculta + '(neta_oculta)') * w_S.T.dot(delta_salida)
            
            w_S = w_S + alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            b_S = b_S + alfa * delta_salida + momento * momento_b_S
             
            w_O = w_O + alfa * delta_oculta * P[:,p] + momento * momento_w_O
            b_O = b_O + alfa * delta_oculta + momento * momento_b_O
           
            momento_w_S = alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            momento_b_S = alfa * delta_salida + momento * momento_b_S
            
            momento_w_O = alfa * delta_oculta * P[:,p].T + momento * momento_w_O
            momento_b_O = alfa * delta_oculta + momento * momento_b_O
            
        error_prom = suma_error / CantPatrones
        ite = ite + 1
        print(ite, error_prom)   
        
        if dibujar and (entran == 2):        
            plot(P, T2, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom))
        
    return (w_O, b_O, w_S, b_S, ite, error_prom)


def train_w_fijos(P, T, T2, ocultas, alfa, momento, fun_oculta, fun_salida, MAX_ITE, cota_error, dibujar,random):
    (entran, CantPatrones) = P.shape
    (salidas, CantPatrones) = T.shape

    if random is None:
        w_O = np.random.rand(ocultas, entran)
        b_O = np.random.rand(ocultas, 1) - 0.5
        w_S = np.random.rand(salidas, ocultas) - 0.5
        b_S = np.random.rand(salidas, 1) - 0.5
    else:
        w_O = np.ones((ocultas, entran)) * random
        b_O = np.ones((ocultas, 1)) * random - 0.5
        w_S = np.ones((salidas, ocultas)) * random- 0.5
        b_S = np.ones((salidas, 1)) * random - 0.5

    momento_w_S = np.zeros(w_S.shape)
    momento_b_S = np.zeros(b_S.shape)
    momento_w_O = np.zeros(w_O.shape)
    momento_b_O = np.zeros(b_O.shape)

    ite = 0;
    ErrorProm = cota_error + 1

    while (ite < MAX_ITE) and (ErrorProm > cota_error):
        suma_error = 0
        for p in range(CantPatrones):
            neta_oculta = w_O.dot(P[:, p][np.newaxis].T) + b_O
            salida_oculta = eval(fun_oculta + '(neta_oculta)')
            neta_salida = w_S.dot(salida_oculta) + b_S
            salida_salida = eval(fun_salida + '(neta_salida)')

            error_ejemplo = T[:, p] - salida_salida.T[0]
            suma_error = suma_error + np.sum(error_ejemplo ** 2)

            delta_salida = error_ejemplo[np.newaxis].T * eval('d' + fun_salida + '(neta_salida)')
            delta_oculta = eval('d' + fun_oculta + '(neta_oculta)') * w_S.T.dot(delta_salida)

            w_S = w_S + alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            b_S = b_S + alfa * delta_salida + momento * momento_b_S

            w_O = w_O + alfa * delta_oculta * P[:, p] + momento * momento_w_O
            b_O = b_O + alfa * delta_oculta + momento * momento_b_O

            momento_w_S = alfa * delta_salida * salida_oculta.T + momento * momento_w_S
            momento_b_S = alfa * delta_salida + momento * momento_b_S

            momento_w_O = alfa * delta_oculta * P[:, p].T + momento * momento_w_O
            momento_b_O = alfa * delta_oculta + momento * momento_b_O

        error_prom = suma_error / CantPatrones
        ite = ite + 1
        print(ite, error_prom)

        if dibujar and (entran == 2):
            plot(P, T2, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(error_prom))

    return (w_O, b_O, w_S, b_S, ite, error_prom)