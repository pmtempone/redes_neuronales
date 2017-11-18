#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:40:21 2017

@author: pablotempone

Title: Actividad 3 - Ejercicio 1
"""
import numpy as np
import pandas as pd
import Ejercicios.Actividad_3.bpn as bpn

drugs = pd.read_excel('/Volumes/Disco_SD/redes_neuronales/Ejercicios/Actividad_3/Drug4.xlsx',header=None)

drugs_np = np.array(drugs)

drugs_np_scale = (drugs_np[:,:6]-drugs_np[:,:6].min(axis=0))/(drugs_np[:,:6].max(axis=0)-drugs_np[:,:6].min(axis=0))

P = drugs_np_scale[:,:6].T
T = drugs_np[:,6]

T_matriz = np.concatenate(([T==0], [T==1], [T==2], [T==3]), axis=0).astype(int)


(w_O, b_O, w_S, b_S, ite, errorProm) = bpn.train_w_fijos(P,T_matriz, T, 3 , 0.5, 0.5, 'logsig', 'logsig', 1000, 0.001, True,-20.0)


