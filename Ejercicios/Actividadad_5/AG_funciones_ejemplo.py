# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: Waldo
       
"""


import numpy as np
import matplotlib.pyplot as plt
import AG

#------------------------------------------------------------------------------
# Parámetros de funciones
parametros_funciones = {1: (25, -1, 5),     # ===== f(x) = x^2 ======
                        2: (3, -2, 1),      # ===== f(x) = -x * sin(10 * x * pi) ======
                        3: (1, -50, 50),    # ===== f(x) = 0.5+((sin( sqrt(x^2) )^2 - 0.5) /(1 + 0.001*(x^2))^2)  ======
                        4: (1, -50, 50)}    # ===== f(x) = sin( sqrt(x^2 +100 ) ) / sqrt( x^2+1)    ======

funcion_elegida = 1 # 1..4
#------------------------------------------------------------------------------

# Parámetros
longitud_cromosoma  = 20        
tamaño_poblacion = 20 
probabilidad_crossover  = 0.75
probabilidad_mutacion = 0.1
cantidad_generaciones = 50
elitismo = True

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Funciones auxiliares
def fun(f, x):
    if(f==1):
        return x**2
    if(f==2):
        return ((-x) * np.sin(10 * np.pi * x) ) + 1
    if(f==3):
        return 0.5 + ((np.sin( np.sqrt(x**2) )**2 - 0.5) / (1 + 0.001 * (x**2))**2)
    if(f==4):
        return np.sin( np.sqrt(x**2 + 100 ) ) / np.sqrt( x**2 + 1)

def graficar(gen, fenotipos = [], mejorFenotipo = None, mejorFitnes = None):
    plt.clf()
    
    x_min = parametros_funciones[funcion_elegida][1]
    x_max = parametros_funciones[funcion_elegida][2]
    
    x = np.array(range(x_min*1000, x_max*1000)) / 1000  
    y = fun(funcion_elegida, x)        

    plt.axis([x_min, x_max, np.min(y), np.max(y)])
    plt.plot(x, y, color='b')
    
    if fenotipos != []:
        fenotipos = fenotipos[0,:]
    
    xx = []
    yy = []
    for i in range(len(fenotipos)):
        xx.append(fenotipos[i])
        yy.append(fun(funcion_elegida, fenotipos[i]))
    plt.scatter(xx, yy, marker='*', color='r')
    
    if(not mejorFenotipo is None):
        plt.title('Gen = {1} - Mejor individuo => x = {0:.5f}'.format(fenotipos[mejorFenotipo], gen))
    
    plt.draw()
    plt.pause(0.00001) 
    
#------------------------------------------------------------------------------

def genotipo2fenotipo(poblacion):
    (long, tama) = poblacion.shape
    ValorMax = 2**long - 1
    Min = parametros_funciones[funcion_elegida][1]
    Max = parametros_funciones[funcion_elegida][2]
    
    potencias = np.array(range(long-1, -1, -1))
    base = 2**potencias
    decimales = sum((poblacion.T * base).T)
    
    return (Min + decimales / ValorMax * (Max-Min))[np.newaxis]

def evaluarFitness(fenotipos):
    y = fun(funcion_elegida, fenotipos) 
    cmax = parametros_funciones[funcion_elegida][0]
    return (abs(cmax - y))[0,:]
    

(individuo, fitness) = AG.evolucionar(longitud_cromosoma, tamaño_poblacion, probabilidad_crossover, 
                           probabilidad_mutacion, cantidad_generaciones, elitismo,
                               genotipo2fenotipo, evaluarFitness, graficar)

print (individuo, fitness)