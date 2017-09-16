# Clase 03 - 

## Adaline (Widrow)

Introdujo la regla delta. Algoritmo LMS (least mean square).
Arquitectura es igual al perceptron.
Toma valores entre 0 y 1. Función continua.

Intenta minimizar el error cuadrático. 

### Regla delta:

+ Elijo valores aleatoreos.
+ La expresión es derivable.

Minimo en w_0 = -1, w_1 = 2

**Alpha** es la velocidad de aprendizaje.

**Siempre escalar las variables**

### Para clasificar dos variables

Se agrega una nueva dimensión separando mis datos.

## Funciones más usadas

+ Purelin: fx lineal
+ Logsis: Sigmoide acotada entre 0 y 1, con derivada ya establecida.
+ Tansig: Sigmoide acotada entre -1 y 1, la derivada es en función de la propia función. Se usa para actualizar los w.
