# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:28:31 2017

@author: auvimo
"""

import numpy as np
import scipy as sp
import PIL
import scipy.misc


def AbrirImagen(archivo):
    datos = np.array(sp.misc.imread(archivo, mode='P'))
    maximo = len(datos)
    X = np.array([], dtype=np.int64).reshape(0,3)
    colores = np.array([0, 9, 12, 10]) # negro rojo azul verde
    for color in colores:
        filas, columnas = np.where(datos == color)
        clase = np.where(colores == color)[0][0] 
        clases = [clase] * len(filas)
        X = np.vstack([X, np.column_stack((columnas+1, maximo-(filas+1), clases))])
    return X


if __name__ == '__main__':
    archivo = r'Imagen.bmp'
    print(AbrirImagen(archivo))
