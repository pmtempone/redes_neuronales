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

#tomar individuo del ejercicio 2: array([-20., -20., -20., -20., -20., -20., -20.])

#decodificar individuo

pesos_a = genotipo2pesos(individuo)

print(pesos_a)

#entrenar con esos pesos como iniciales

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train_w_fijos(P,T_matriz, T, 3 , 0.1, 0.05, 'logsig', 'tansig', 1000, 0.001, True,2)
