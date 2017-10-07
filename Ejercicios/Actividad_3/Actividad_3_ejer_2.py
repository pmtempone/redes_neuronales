#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Sep 30 11:40:21 2017

@author: pablotempone

Title: Actividad 3 - Ejercicio 2
"""
import numpy as np
import Ejercicios.Actividad_3.ImagenPuntos as ip
import PIL
from scipy import misc
import Ejercicios.Actividad_3.bpn as bpn

figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 1.bmp')
figura2 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 2.bmp')
figura3 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 3.bmp')
figura4 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 4.bmp')
figura5 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Figura 5.bmp')

#Imagen 1

figura1_scale = (figura1[:,:2]-figura1[:,:2].min(axis=0))/(figura1[:,:2].max(axis=0)-figura1[:,:2].min(axis=0))

P = figura1_scale[:,:2].T
T = figura1[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T, 1, 0.1, 0.05, 'logsig', 'tansig', 1000, 0.001, True)

#Imagen 2

figura2_scale = (figura2[:,:2]-figura2[:,:2].min(axis=0))/(figura2[:,:2].max(axis=0)-figura2[:,:2].min(axis=0))

P = figura2_scale[:,:2].T
T = figura2[:,2]

T_matriz = np.concatenate(([T==0], [T==1],[T==2]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T, 3, 0.2, 0.05, 'logsig', 'tansig', 1000, 0.005, True)

#Imagen 3

np.random.shuffle(figura3)
figura3_scale = (figura3[:,:2]-figura3[:,:2].min(axis=0))/(figura3[:,:2].max(axis=0)-figura3[:,:2].min(axis=0))

P = figura3_scale[:,:2].T
T = figura3[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T,4, 0.2, 0.1, 'logsig', 'logsig', 1000, 0.005, True)

#Imagen 4

np.random.shuffle(figura4)
figura4_scale = (figura4[:,:2]-figura4[:,:2].min(axis=0))/(figura4[:,:2].max(axis=0)-figura4[:,:2].min(axis=0))

P = figura4_scale[:,:2].T
T = figura4[:,2]

T_matriz = np.concatenate(([T==0], [T==1],[T==2], [T==3]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T,4, 0.3, 0.2, 'logsig', 'logsig', 1500, 0.005, False)

bpn.plot(P, T, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(errorProm))

#Imagen 5

np.random.shuffle(figura5)
figura5_scale = (figura5[:,:2]-figura5[:,:2].min(axis=0))/(figura5[:,:2].max(axis=0)-figura5[:,:2].min(axis=0))

P = figura5_scale[:,:2].T
T = figura5[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T,5, 0.3, 0.2, 'logsig', 'logsig', 1500, 0.005, True)

bpn.plot(P, T, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(errorProm))

#punto c, triple de neuronas

#Imagen 5

np.random.shuffle(figura5)
figura5_scale = (figura5[:,:2]-figura5[:,:2].min(axis=0))/(figura5[:,:2].max(axis=0)-figura5[:,:2].min(axis=0))

P = figura5_scale[:,:2].T
T = figura5[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)

(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train(P,T_matriz, T,5*3, 0.3, 0.2, 'logsig', 'logsig', 1500, 0.005, False)

bpn.plot(P, T, w_O, b_O, 'Iteración: ' + str(ite) + ' - Error promedio: ' + str(errorProm))