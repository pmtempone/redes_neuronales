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
import cpn

figura1 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 1.bmp')
figura2 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 2.bmp')
figura3 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 3.bmp')
figura4 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 4.bmp')
figura5 = ip.AbrirImagen('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Act_4/Figura 5.bmp')

#Imagen 1

figura1_scale = (figura1[:,:2]-figura1[:,:2].min(axis=0))/(figura1[:,:2].max(axis=0)-figura1[:,:2].min(axis=0))

P = figura1_scale[:,:2].T
T = figura1[:,2]

T_matriz = np.concatenate(([T==0], [T==1]), axis=0).astype(int)
