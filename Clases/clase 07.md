# Clase 07: Algorítmos genéticos

## Características

+ Se utiliza para la búsqueda de una solución. Ej: encontrar el equilibro en la temperatura de un aire acondicionado.
+ No se puede saber si es la mejor si tiene infinitas soluciones.
+ Se usan cuando los problemas no se pueden representar con ecuaciones matemáticas. Ejemplo: caja de cartón, encontrar el r para doblar la caja.
+ funcionan con el azar
+ Estan formados por individuos, que son cada posible solución.
    + Aptos
    + Malos

## Funcionamiento

Algoritmos se combinan para encontrar otra solución.
    + Mutación
    + Combinación

Se seleccionan los más aptos.
    + Función de evaluación de aptitud.
    + Valores de los parámetros: tamaño de población, fitnes, etc.

## Representación

+ Cromosomas: un arreglo de valores
+ Problema: minimizar o maximizar una función
+ Holland: cuanto más largo es el cromosoma, es más fácil almacenar el cromosoma.
+ Genotipos: representación de la solución. En una cadena de valores binarios.
+ Fenotipos: nro que sirve, transformación del genotipo.

## Problemas

+ A veces si la solución es muy grande, encontrar la representación genética a esa solución es muy costosa.
+ Soluciones ilegales

## Códigos de Gray

## Función de aptitud

+ Dado un genotipo, se determina el fenotipo
+ Método de la ruleta, permite definir la probabilidad de que un valor de ser seleccionado.
    + Puede generar que los padres se repitan
+ Método de torneo

## CrossOver

+ Se eligen las parejas.
+ Se arma el punto de cruce. Es seleccionado al azar.
+ Se recombina para obtener los hijos.
+ Para que se haga crossover se utiliza la probabilidad, generada al azar para caso.

## Mutación

Mutación: sirve para que aparezcan arreglos donde no los hay. Ej: todos los padres tienen 0 en el 3er gen, si se hace sobre ese gen nunca va a cambiar, con mutación se aplica un cambio poniendo un 1. 
    + Se aplica gen a gen.

## Elitismo

## Evaluación


# Neuroevolución

Se evolucionan los pesos finales de una red neuronal ya entrenada, esto permite mejorar la red sin tener que reentrenearla toda.

## Función de fitness

Error promedio: Más chico el error, más grande el fitness (más apto).

