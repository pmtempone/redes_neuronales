# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: Waldo
       
"""


import numpy as np
import matplotlib.pyplot as plt
import Ejercicios.Actividadad_5.AG as AG
import numpy as np
import Ejercicios.Actividad_3.ImagenPuntos as ip
import PIL
from scipy import misc
import Ejercicios.Actividad_3.bpn as bpn

#------------------------------------------------------------------------------
# A COMPLETAR POR EL ALUMNO

# Lectura de los Datos en P y T para la evaluación del fitness de cada individuo (RN)

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Parámetros
intervalo_pesos = (-20, 25)        # Intervalos donde se representa los pesos
cantidad_pesos_total = 7        # Cantidad de pesos en toda la arquitectura
genes_pesos = 0.05                  # Presición de cada uno de los pesos
longitud_cromosoma = genes_pesos * cantidad_pesos_total

tamaño_poblacion = 40 
probabilidad_crossover  = 0.75
probabilidad_mutacion = 0.011
cantidad_generaciones = 50
elitismo = True

#------------------------------------------------------------------------------

# Funciones auxiliares

# La función graficar puede ser usada para visualizar el mejor individuo
def graficar(gen, fenotipos = [], mejorFenotipo = None, mejorFitnes = None):
    pass

def EvaluarFitnessDeRN(pesos):
    # pesos es un vector con todos los pesos de la red
    # deberá decodificar este vector armando las matrices
    # correspondientes a la arquitectura elegida para poder
    # evaluar el individuo (red neuronal)
    w_O = np.matrix(pesos[:2])
    b_O = np.matrix(pesos[2])
    w_S = np.matrix(pesos[3:5]).T
    b_S = np.matrix(pesos[5:]).T
    print('termine variables')
    # Como valor de fitness deberá devolver el accuracy
    # alcanzado al clasificar los puntos de la matriz de P
    error = errorCuadMedio(w_O, b_O, w_S, b_S)
    error_cuad = 1/(1+error)
    return error_cuad

def errorCuadMedio(w_O,b_O,w_S,b_S):
    print('arranco loop')
    #levanto variables que necesito
    figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 1.bmp')
    figura1_scale = (figura1[:, :2] - figura1[:, :2].min(axis=0)) / (
    figura1[:, :2].max(axis=0) - figura1[:, :2].min(axis=0))
    P = figura1_scale[:, :2].T
    T = figura1[:, 2]
    T_matriz = np.concatenate(([T == 0], [T == 1]), axis=0).astype(int)
    (entran, CantPatrones) = P.shape
    (salidas, CantPatrones) = T_matriz.shape
    print('arranco loop')
    fun_oculta = 'logsig'
    fun_salida = 'tansig'
    suma_error = 0
    for p in range(CantPatrones):
        neta_oculta = w_O.dot(P[:, p][np.newaxis].T) + b_O
        salida_oculta = eval(fun_oculta + '(neta_oculta)')
        neta_salida = w_S.dot(salida_oculta) + b_S
        salida_salida = eval(fun_salida + '(neta_salida)')
        error_ejemplo = T_matriz[:, p] - salida_salida.T[0]
        print(error_ejemplo)
        print(p)
        suma_error = suma_error + np.sum(np.array(error_ejemplo) ** 2)
    print('error calculo')
    error_prom = suma_error / CantPatrones
    print(error_prom)

    return error_prom


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
#------------------------------------------------------------------------------

def genotipo2pesos(individuo):
    ValorMax = 2**genes_pesos
    k = 0  # Posicion del proximo peso a convertir
    w = np.zeros(cantidad_pesos_total)
    for j in range(cantidad_pesos_total):
        print(j)
        peso_binario = individuo[int(k) : int(k + genes_pesos)]
        
        potencias = np.array(range(int(genes_pesos)-1, -1, -1))
        base = 2**potencias
        peso_decimal = sum(peso_binario * base)
        peso_real = intervalo_pesos[0] + (peso_decimal / ValorMax) * (intervalo_pesos[1] - intervalo_pesos[0])
        
        w[j] = peso_real 
        k = k + genes_pesos
    return w

def genotipo2fenotipo(poblacion):
    (long, tama) = poblacion.shape
    
    y = np.zeros((cantidad_pesos_total, tama));
    for i in range(tama):
        individuo = poblacion[:,i]
        y[:,i] = genotipo2pesos(individuo);
    return y

def evaluarFitness(fenotipos):
    (long, tama) = fenotipos.shape
    
    y = np.zeros(tama);
    for i in range(tama):
        pesos = fenotipos[:,i]
        y[i] = EvaluarFitnessDeRN(pesos);
    return y
    

#(individuo, fitness) = AG.evolucionar(longitud_cromosoma, tamaño_poblacion, probabilidad_crossover,
#                           probabilidad_mutacion, cantidad_generaciones, elitismo,
#                               genotipo2fenotipo, evaluarFitness, graficar)

#print (individuo, fitness)


#poblacion = np.round(np.random.rand(longitud_cromosoma, tamaño_poblacion))
#fenotipos = genotipo2fenotipo(poblacion)
#fitness = evaluarFitness(fenotipos)
#mejor = np.argmax(fitness)
#fenotipos = genotipo2fenotipo(poblacion)
#print( fenotipos[mejor], fitness[mejor])
#graficar(fenotipos)