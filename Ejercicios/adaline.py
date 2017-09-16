# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: auvimo

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
	   CotaError: cota de error tolerable
	   funcion: 'purelin', 'logsig' o 'tansig'   
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y la recta discriminante.

Devuelve:
       W: la matriz de pesos W del percpetrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
       ite: número de iteraciones ejecutadas durante el algoritmo. Si
           devuelve el mismo valor que MAX_ITERA es porque no pudo finalizar
           con el entrenamiento
       ErrorProm: error promedio obtenido
		

Ejemplo de uso:
	   (W, b, ite, ErrorProm) = adaline.train(P, T, 0.2, 10, 0.0000001, 'purelin', False)
       
*******************************************************************************
       
Función plot
-------------       
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar el perceptrón. Los ejemplos deben estar en columnas.
       T: es un vector fila con la clase esperada para cada ejemplo. Los
           valores de las clases deben ser 0 (cero) y 1 (uno)
       W: la matriz de pesos W del perceptrón entrenado
       b: valor del bias (W0) del perceptrón entrenado
	   title: título de la figura

Ejemplo de uso:
       plot(P, T, W, b, title);


"""

import numpy as np
import matplotlib.pyplot as plt

def plot(P, T, W, b, title):
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
    
    plt.title(title)
    
    plt.draw()
    plt.pause(0.00001) 
    
def purelin(x):
    #return x.copy()
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
    
def train(P, T, alfa, MAX_ITE, CotaError, funcion, dibujar):
    (entran, CantPatrones) = P.shape    
    W = np.random.rand(1, entran)
    b = np.random.rand()
    
    T2 = T.copy();
    T2 = np.floor((T2 + 1) / 2)

    ite = 0;
    ErrorProm = 1
    
    while (ite < MAX_ITE) and (ErrorProm > CotaError):
        SumaError = 0
        for p in range(CantPatrones): 
           neta = b + W.dot(P[:, p])
           
           salida = eval(funcion + '(neta)')            
           errorPatron = T[p] - salida
           SumaError = SumaError + errorPatron ** 2
           
           derivada = eval('d' + funcion + '(neta)')
           
           grad_b = -2 * errorPatron * derivada;
           grad_W = -2 * errorPatron * derivada * P[:, p]

           b = b - alfa * grad_b;
           W = W - alfa * grad_W;         
        ErrorProm = SumaError / CantPatrones
        ite = ite + 1
        print(ite, ErrorProm)   
        
        if dibujar and (entran == 2):        
            plot(P, T2, W, b, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(ErrorProm))
        
    return (W, b, ite, ErrorProm)