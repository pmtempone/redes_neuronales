# Clase 06

## redes de contrapropagación: Características

+ Primero se entrena capa competitiva y luego se entrena la capa de salida
+ Es más rápida de entrenar
+ Muchas veces el resultado no es el deseado.
+ Neurona ganadora va a ser aquella que sea más similar es la ganadora, esa va a adaptar su vector de pesos. La que tenga mayor estímulo.
+ Neuronas tratan de aprender de aquellos patrones de los que gana. Serían neuronas especialistas en determinados casos.
+ Es similar al clustering, todas las neuronas arrancan con mismos valores y cada una se va ajustando a los datos.
+ No es muy útil para clases desbalanceadas.
+ Se premia o castiga a las neuronas con el bias
+ El w final tiende a 1/0 o 1/-1 y eso permite saber que tantos casos no puede predecir.

## SOM: Características

+ No supervisado
+ Las neuronas intercambian entre si información
+ El mapa se adapta a la topología de los datos