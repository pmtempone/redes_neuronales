# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:06:07 2017

@author: Waldo
"""

import numpy as np

def ruleta(fitness):
    cuantos = len(fitness)
    suma_fitness = sum(fitness)
    aleatorio = np.random.rand() * suma_fitness   # posicion dentro de la ruleta

    # buscando el slot correspondiente
    suma = fitness[0]
    j = 0
    while (suma < aleatorio) and (j < (cuantos-1)):
        j = j + 1
        suma = suma + fitness[j]

    return j

def mutar(alelo, probabilidad):
    # muta el ALELO con la probabilidad inidicada
    if ((probabilidad == 1) or (np.random.rand() <= probabilidad)):
        # hay mutacion
        return not alelo
    else:
        return alelo

def cruzar_y_mutar(padre1, padre2, probabilidad_crossover, probabilidad_mutacion):
    long = len(padre1)
    # ver si corresponde aplicar crossover
    hayCrossover = ((probabilidad_crossover == 1) or (np.random.rand() <= probabilidad_crossover))
    if (hayCrossover):
        posicion = round(np.random.rand() * (long-2)) + 1
    else:
        posicion = long

    hijo1 = np.zeros(long)
    hijo2 = np.zeros(long)
    for i in range(posicion):
        hijo1[i] = mutar(padre1[i], probabilidad_mutacion)
        hijo2[i] = mutar(padre2[i], probabilidad_mutacion)

    if (hayCrossover):
        for i in range(posicion, long):
            hijo1[i] = mutar(padre2[i], probabilidad_mutacion)
            hijo2[i] = mutar(padre1[i], probabilidad_mutacion)

    return (hijo1, hijo2)

def generar(poblacion, fitness, probabilidad_crossover, probabilidad_mutacion):
    (long, cinds) = poblacion.shape   
    parejas = round(cinds / 2) # por cada iteracion se generan dos hijos

    nueva = np.zeros((long, cinds))
    for i in range(parejas):
        # seleccionar dos padres
        padre1 = ruleta(fitness)
        padre2 = ruleta(fitness)
        
        #generar los dos hijos
        (hijo1, hijo2) = cruzar_y_mutar(poblacion[:, padre1], poblacion[:, padre2], probabilidad_crossover, probabilidad_mutacion)
    
        nueva[:, i*2] = hijo1
        nueva[:, i*2 + 1] = hijo2
    
    return nueva

def evolucionar(longitud_cromosoma, tamaño_poblacion, probabilidad_crossover, 
               probabilidad_mutacion, cantidad_generaciones, elitismo, 
               funcion_fenotipos, funcion_evaluar_fitness, funcion_graficar = None):
    
    # Construir la poblacion inicial
    poblacion = np.round(np.random.rand(longitud_cromosoma, tamaño_poblacion))
    fenotipos = funcion_fenotipos(poblacion)
    fitness = funcion_evaluar_fitness(fenotipos)
    
    mejor = np.argmax(fitness)
    mejor_fitness = fitness[mejor]
    mejor_individuo = poblacion[:, mejor]
    mejor_fenotipo = fenotipos[:,mejor]
    
    gen = 1
    while (gen <= cantidad_generaciones):        
        m = np.argmax(fitness)
        if(fitness[m] > mejor_fitness):
            mejor = m
            mejor_fitness = fitness[mejor]
            mejor_individuo = poblacion[:, mejor]
            mejor_fenotipo = fenotipos[:,mejor]
   
        poblacion = generar(poblacion, fitness, probabilidad_crossover, probabilidad_mutacion)
        fenotipos = funcion_fenotipos(poblacion)
        fitness = funcion_evaluar_fitness(fenotipos)

        if (elitismo):
            peor = np.argmin(fitness)
            poblacion[:, peor] = mejor_individuo
            fitness[peor] = mejor_fitness
            fenotipos[:,peor] = mejor_fenotipo
            
            mejor = peor

        if (not (funcion_graficar is None)):
            funcion_graficar(gen, fenotipos, mejor, mejor_fitness)
   
        # fitness maximo y promedio
        FitAVG = sum(fitness) / tamaño_poblacion
        print (gen, mejor_fitness, FitAVG)
        gen = gen + 1
        
    return mejor_fenotipo, mejor_fitness
