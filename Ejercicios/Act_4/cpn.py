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
       alfa y beta: velocidad de aprendizaje de la capa oculta y de salida respectivamente.
	   gamma: parámetro utilizado para el cálculo del bias.
       max_ite_O y max_ite_S: la cantidad máxima de iteraciones en las cuales se va a
           entrenar la capa oculta y de salida respectivamente.
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y los centroides.

Devuelve:
       w_O: la matriz de pesos de las neuronas de la capa oculta
       w_S: la matriz de pesos de las neuronas de la capa de salida

Ejemplo de uso:
	   (w_O, w_S) = train(P, T, T2, ocultas, alfa, beta, gamma, max_ite_O, max_ite_S, dibujar)
       
"""


import numpy as np
import matplotlib.pyplot as plt

marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm')}

def plot(P, T, W, title):
    plt.clf()
    
    #Ejemplos
    colores = len(marcadores)
    for class_value in np.unique(T):
        x = []
        y = []
        for i in range(len(T)):
            if T[i] == class_value:
                x.append(P[0, i])
                y.append(P[1, i])
        plt.scatter(x, y, marker=marcadores[class_value % colores][0], color=marcadores[class_value % colores][1])
    
    #ejes
    minimos = np.min(P, axis=1)
    maximos = np.max(P, axis=1)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #centroides
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        plt.scatter(W[neu,0], W[neu,1], marker='o', color='r')
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001)
    
def train(P, T, T2, ocultas, alfa, beta, gamma, max_ite_O, max_ite_S, dibujar):
    (entran, CantPatrones) = P.shape   
    (salidas, CantPatrones) = T.shape   
    
    w_O = np.random.rand(ocultas, entran) - 0.5
    w_S = np.random.rand(salidas, ocultas) - 0.5
    
    w_O = np.ones((ocultas, entran)) * 0.5
    
    c = np.ones((1,ocultas)) / ocultas;
    b = np.exp(1 - np.log(c));
    
    ite = 0;
    
    while (ite < max_ite_O):
        for p in range(CantPatrones): 
            distancias = -np.sqrt(np.sum((w_O-(P[:,p][np.newaxis])*np.ones((ocultas,1)))**2,1)) + b
            ganadora = np.argmax(distancias)
           
            w_O[ganadora,:] = w_O[ganadora,:] + alfa * (P[:,p] - w_O[ganadora,:])
            
            #recalculo del bias
            a = np.zeros((1, ocultas))
            a[0, ganadora] = 1
            
            c = np.exp(1-np.log(b))
            c = (1-gamma) * c + gamma * a
            
            b = np.exp(1- np.log(c)) - gamma * b
            
        ite = ite + 1
        
        if dibujar and (entran == 2):        
            plot(P, T2, w_O, 'Iteración: ' + str(ite))
        
    ite = 0;
    T3 = T2
    while ( ite <= max_ite_S ):
        for p in range(CantPatrones): 
            distancias = -np.sqrt(np.sum((w_O-(P[:,p][np.newaxis])*np.ones((ocultas,1)))**2,1))
            ganadora = np.argmax(distancias)
       
            w_S[:, ganadora] = w_S[:, ganadora] + beta * (T[:, p] - w_S[:, ganadora])
            T3[p] = ganadora
    
        ite = ite + 1
    plot(P, T3, w_O, 'Fin')
    return (w_O, w_S)