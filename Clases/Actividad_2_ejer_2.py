#Actividad 2 - Ejercicio 2
#=========================

import pandas as pd
import numpy as np

radar = pd.read_excel("/Users/pablotempone/Google Drive/Maestria/Redes Neuronales/Ejercicio2/radar.xlsx")

#a.escalar variables

radar -= radar.min()  # equivalent to df = df - df.min()

radar /= radar.max()  # equivalent to df = df / df.max()

#b.elija 3 valores de alfa



