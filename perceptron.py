# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:49:37 2017

@author: Waldo

Función train
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en columnas.
       T: es un vector fila con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       alfa: velocidad de aprendizaje
       MAX_ITERA: la cantidad máxima de iteraciones en las cuales se va a
           ejecutar el algoritmo
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y la recta discriminante.

Devuelve:
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       ite: número de iteraciones ejecutadas durante el algoritmo. Si
           devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
           con el entrenamiento

Ejemplo de uso:
       [W, b, ite] = train(P, T, 0.25, 250, True);
       
*******************************************************************************
       
Función plot
-------------       
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en columnas.
       T: es un vector fila con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado

Ejemplo de uso:
       plot(P, T, W, b);


"""

import numpy as np
import matplotlib.pyplot as plt

def plot(P, T, W, b):
    plt.clf()
    
    #ceros
    x=[]
    y=[]
    for i in range(len(T)):
        if T[i] == 0:
            x.append(P[0, i])
            y.append(P[1, i])
    plt.scatter(x, y, marker='+', color='b')
    
    #unos
    x=[]
    y=[]
    for i in range(len(T)):
        if T[i] == 1:
            x.append(P[0, i])
            y.append(P[1, i])
    plt.scatter(x, y, marker='o', color='g')
    
    #ejes
    minimos = np.min(P, axis=1)
    maximos = np.max(P, axis=1)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #recta discriminante
    m = W[0,0] / W[0,1] * -1
    n = b / W[0,1] * -1
    x1 = minimos[0]
    y1 = x1 * m + n
    x2 = maximos[0]
    y2 = x2 * m + n
    plt.plot([x1, x2],[y1, y2], color='r')
    
    plt.draw()
    plt.pause(0.00001)   

def train(P, T, alfa, max_itera, dibujar):
    (entran, cantPatrones) = P.shape
    W = np.random.rand(1,entran)
    b = np.random.rand()
    
    ite = 0
    otra_vez = True
    
    plt.ion()
    plt.show()
    
    while ((ite <= max_itera) and otra_vez):
        otra_vez = False
        ite = ite + 1
        
        for patr in range(cantPatrones):
            salida = b + W.dot(P[:, patr]) 
            if salida >= 0:
                salida = 1
            else:
                salida = 0
      
            factor = alfa * (T[patr] - salida)
            if (factor != 0):
                otra_vez = True
                W = W + factor * P[:, patr]
                b = b + factor
        
        if dibujar and (entran == 2):        
            plot(P, T, W, b)            

    if dibujar and (entran == 2):        
        plot(P, T, W, b)
        
    return (W, b, ite)