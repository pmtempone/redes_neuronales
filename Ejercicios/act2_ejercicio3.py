import matplotlib.pyplot as plt
from scipy.io import wavfile as wf
from Ejercicios.adaline import train

import numpy as np
%matplotlib inline
samplerate, audioOriginal = wf.read('/Users/pablotempone/Google Drive/Maestria/Redes Neuronales/Ejercicio2/entrevista.wav')

plt.figure();
plt.plot(audiooriginal)

max_col=np.max(audioOriginal, axis=0)
min_col=np.min(audioOriginal, axis=0)
dif_col=max_col-min_col

audioOriginalEscalado = (audioOriginal - min_col) / dif_col

#%%

r=0.7
numRamdom = np.random.uniform(0,1, size=len(audioOriginalEscalado))  - 0.5
audioRuidoEscalado = audioOriginalEscalado +numRamdom *r

#%%
w = 6

def rolling_window(a, window):
   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
   strides = a.strides + (a.strides[-1],)
   return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

rwOriginalEscalado = rolling_window(audioOriginalEscalado, w+1)
rwRuidoEscalado = rolling_window(audioRuidoEscalado, w+1)

target = rwOriginalEscalado[:,3]

#%%
alpha=0.05
MAXITERATIONS=300
cota_error= 0.0001
funcionSalida= 'purelin'

rwRuidoEscalado_t =np.transpose(rwRuidoEscalado)
target_t = np.transpose(target)
#target_t = (target)

muestras = 8000
rwRuidoEscalado_t_TRAIN =rwRuidoEscalado_t [: ,0:muestras]
target_t_TRAIN = target_t [0:muestras]

tuplaAudio = train (rwRuidoEscalado_t_TRAIN, target_t_TRAIN, alpha, MAXITERATIONS, cota_error, funcionSalida,True)

#%%
a = tuplaAudio[0] * rwRuidoEscalado_t_TRAIN
#%%
tuplaAudio[1].shape

#%%
audioReconstruido = np.dot(tuplaAudio[0],rwRuidoEscalado_t)+ tuplaAudio[1]



