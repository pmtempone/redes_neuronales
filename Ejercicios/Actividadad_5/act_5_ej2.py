# -*- coding: utf-8 -*-

import numpy as np
import Ejercicios.Actividad_3.ImagenPuntos as ip
import PIL
from scipy import misc
import Ejercicios.Actividad_3.bpn as bpn
from Ejercicios.Actividadad_5.AG import evolucionar
from Ejercicios.Actividadad_5.RNE import errorCuadMedio, EvaluarFitnessDeRN,genotipo2fenotipo,genotipo2pesos,evaluarFitness,graficar
import Ejercicios.Actividadad_5.RNE as RNE
import Ejercicios.Actividadad_5.AG as AG


figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 1.bmp')

figura1_scale = (figura1[:,:2]-figura1[:,:2].min(axis=0))/(figura1[:,:2].max(axis=0)-figura1[:,:2].min(axis=0))

P = figura1_scale[:,:2].T
T = figura1[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

intervalo_pesos = (-20, 25)        # Intervalos donde se representa los pesos
cantidad_pesos_total = int(7)        # Cantidad de pesos en toda la arquitectura
genes_pesos = int(10)                 # Presición de cada uno de los pesos
longitud_cromosoma = genes_pesos * cantidad_pesos_total

tamaño_poblacion = 40
probabilidad_crossover  = 0.75
probabilidad_mutacion = 0.011
cantidad_generaciones = 50
elitismo = True



(individuo, fitness) = evolucionar(longitud_cromosoma, tamaño_poblacion, probabilidad_crossover,probabilidad_mutacion, cantidad_generaciones, elitismo,genotipo2fenotipo, evaluarFitness, graficar)

individuo

fitness