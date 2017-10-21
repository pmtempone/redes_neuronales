# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: auvimo

Función train
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar la red neuronal. Los ejemplos deben estar en columnas. 
	    filas: cantidad de filas del mapa.
       columnas: cantidad de columnas del mapa. Junto con filas determinan 
           el tamaño de la red (cantidad de neuronas).
       alfaInicial: velocidad de aprendizaje inicial.
       vecindad: tamaño inicial de la vecindad.
       func_vecindad: función utilizada para la actualización de la vecindad
           (1 lineal, distinto de 1 sombrero).
       sigma: ancho de la función sombrero.
       ite_reduce: cantidad de iteraciones que deben ocurrir para reducir la 
           vecindad.
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y el mapa.

Devuelve:
       w_O: la matriz de pesos de las neuronas de la capa oculta

Ejemplo de uso:
       (w_O) = train(P, filas, columnas, alfaInicial, vecindad, func_vecindad, sigma, ite_reduce, dibujar):
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

marcadores = {0:('+','b'), 1:('o','g'), 2:('x', 'y'), 3:('*', 'm')}

def dendograma(matriz, T):
    (filas, columnas) = matriz.shape
    labels=[]
    for f in range(filas):
        labels.append(str(int(T[f])))
        for c in range(columnas):
            if(f==c):
                matriz[f,c] = 0
            else:
                if(matriz[f,c] == 0):
                    matriz[f,c] = 2
                else:
                    matriz[f,c] = 1 / matriz[f,c]
                    
    
    
    dists = squareform(matriz)
    linkage_matrix = linkage(dists, "single")
    dendrogram(linkage_matrix, labels=labels)
    plt.title("Dendograma")
    plt.show()

def plot(P, W, filas, columnas, pasos, title):
    plt.clf()
    
    #Ejemplos
    x = []
    y = []
    for i in range(P.shape[1]):
        x.append(P[0, i])
        y.append(P[1, i])
    plt.scatter(x, y, marker='+', color='b')
    
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
        
    #conexiones
    for f in range(filas):
        for c in range(columnas):
            n1 = f*columnas + c
            for ff in range(filas):
                for cc in range(columnas):
                    if(pasos[f, c, ff, cc] == 1):
                        n2 = ff*columnas + cc
                        plt.plot([W[n1, 0], W[n2, 0]], [W[n1, 1], W[n2, 1]], color='r')                   
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001)
    
def linkdist(filas, columnas):
    pasos = np.zeros((filas, columnas, filas, columnas))
    for f in range(filas):
        for c in range(columnas):
            for ff in range(filas):
                for cc in range(columnas):
                    pasos[f, c, ff, cc] = abs(f-ff) + abs(c-cc)
    return pasos
    
def train(P, filas, columnas, alfaInicial, vecindad, func_vecindad, sigma, ite_reduce, dibujar):
    (entran, CantPatrones) = P.shape   

    ocultas = filas * columnas    
    w_O = np.random.rand(ocultas, entran) - 0.5
    
    #w_O = np.ones((ocultas, entran)) * 0
    
    pasos = linkdist(filas, columnas)
    
    max_ite = ite_reduce * (vecindad + 1)
    ite = 0;
    
    while (ite < max_ite):
        alfa = alfaInicial * (1 - ite / max_ite)
        for p in range(CantPatrones): 
            distancias = -np.sqrt(np.sum((w_O-(P[:,p][np.newaxis])*np.ones((ocultas,1)))**2,1))
            ganadora = np.argmax(distancias)
            fila_g = int(np.floor(ganadora / columnas))
            columna_g = int(ganadora % columnas)

            for f in range(filas):
               for c in range(columnas):
                   if(pasos[fila_g, columna_g, f, c] <= vecindad):
                       if func_vecindad == 1:
                           gamma = 1
                       else:
                           gamma = np.exp(- pasos[fila_g, columna_g, f, c] / (2*sigma))
              
                       n = f * columnas + c
                       w_O[n,:] = w_O[n,:] + alfa * (P[:,p] - w_O[n,:]) * gamma
            
        ite = ite + 1
        
        if (vecindad >= 1) and ((ite % ite_reduce)==0):
            vecindad = vecindad - 1;
        
        if dibujar and (entran == 2):        
            plot(P, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))
        
    return (w_O)