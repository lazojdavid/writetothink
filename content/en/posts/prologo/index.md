---
title: Machine Learning, en retrospectiva. 
date: 2024-10-01
author: David Vasquez
draf : false
description: Introducción 
tags: Machine Learning

---
>*Los límites de mi lenguaje son los límites de mi mundo" Ludwig Wittgenstein Filósofo y matemático*

<div style="text-align: justify;">


## Introducción

En esta introducción mencionaré los conceptos que considero fundamnetales para una tener una perspectiva general,  de aquellos engrajanes con los que se mueve esta maquinaria , además ilustaré la explicación con un ejemplo y se verán los conceptos aplicados en usos practicos a través de proyectos de mis proyectos personales: <a>Housing Price California</a>, <a>o la creación de un modelo para detectar anomalías en el fraude bancario</a>. 
Cada concepto que requiera una explicación en sí, tendrá su espacio aparte en el que explicaré con mas detalle el punto que toque y de esta manera el lector pueda ir complementando la información por conceptos, ofreciendo de esta manera lo que llamo *Grados de conocimiento - de menos más*. Considerandolo una de las formas de aprendizaje que mas me han servido como autodidacta. Dicho esto empezemos.

Utilizaré tres letras clave para poder explicar correctamente las principales relaciones: 

- E : Expericiencia obtenida de la data.
- T : Problemas a resolver. 
- P : Medida de rendimiento del modelo.

El objetivo de esta mécanica es utilizar un lenguaje no técnico al momento de explicar lo que es Machine Learning, con el fin de lograr conceptualizar todo este panorama, sin dejar de lado las convenciones, ya que estas son usadas a lo largo de todos los proyectos y las lecturas que pueden encontrar en diferentes fuentes sobre esta área de conocimiento. Por ende mi intención es que el lenguaje sea lo más claro y sencillo posible, sin dejar de lado términos para una correcta base, permitiendoles unir puntos y ver las relaciones entre todos los concpetos que al principio puede parecer no tener nada en común.


## Machine Learning

Un programa aprende de cierta manera **E** con respecto al problema que se necesita resolver **T**, bajo una medida de rendimiento **P** en la resolución de cierto problema **T**; la eficiencia con la que, en base a su experiencia resuelve dicho problema va mejorando a su vez con la cantidad de información brindada. La información brindada proviene de la data con la cual se puede trabajar.

Por el momento nos quedaremos con este concepto sencillo, que iré enriquenciendo conforme avancemos, *-menos a más-* para que vayan entiendo como es que se juntan las ideas.


### Ejemplo Spam ###

Un caso muy común es el del problema de detectar que correo es spam o no **T**, tendría que pasarle data de correos electrónicos que incluyan correos ordinarios y spam. Con el objetivo que el programa comprenda bajo que patrones han sido señalados como "spam" ciertos correros en esta data, aprendiendo así de ella **E**, pero no de toda la data en total, si no por medio de subcojuntos : <a>training set</a> y <a>test set</a>, el primero sirve para entrenar al modelo y el segundo como métrica de rendimiento **P** , es decir, que tanto aprendió de los datos de entrenamiento *training set* y con que éxito pudo predecir los nuvos datos comparandolos con los datos del *test set*. De esta manera es como funcionaría un programa para detectar spam utilizando Machine Learning.

De manera tradicional se tendrían que programar manualmente las reglas que detecten que correo es spam cada vez que se descubra un nuevo patrón, lo que es una pésima idea. Imaginate tener que estar reescribiendo un código cada vez que descubras que tipo de patrones sigue un correo spam. Sería absurdo. Sería ineficiente he inacabable.
 
<div style="text-align: center;">

![Landscape](/machineretrospectiva/spam1.png)
<figcaption>Fig 1. Detector de spam con programación clásica</figcaption>

</div>

Utilizando Machine Learning la historia es mucho más sencilla de lo que parece. Debido a que la programación se hace de manera automática y este es el poder que nos ofrece debido a la versatilidad de las herramientas que contiene. En este ejemplo, un detector de spam, aprende conforme la información que se le dé, esta puede ser brindada de forma estática dentro de un archivo local <a>batch learning</a> o recopilada en tiempos real <a>online learning. </a> 


<div style="text-align: center;">

![Landscape](/machineretrospectiva/spam2.png)
<figcaption>Fig 2. Detector de spam utilizando Machine Learning</figcaption>



<div style="text-align: justify;">

Como nos vamos dando cuenta este enfoque permite reducir problemas que serían complejos con los paradigmas de programación tradicional. Volviendolos mas sencillos de resolver y de entender. Otra ventaja es que la aplicación de esta técnica nos permite poder aprender en el transcurso de su implementación debido a la cantidad de información *insights* que nos brindará durante la marcha.

- Hay varios <a>Sistemas de Machine learning</a> *enfoques* que dependerá del tipo de data disponible. 
- Cada sistema tiene sus propias herramientas *algoritmos* para la resolución de problemas. 
- Tu principal tarea es escoger los <a>algoritmos de Machine Learning</a> y entrenarlo en base a una data. Puedes fallar en dos cosas: Seleccionar mal un algoritmo o tener una mala data. Esto es lo que separa a los buenos de los malos Ingenieros de Machine Learning.

Es aquí donde separamos la línea entre la necesidad de tener herramientas conceptuales claras. Hasta ahora hemos visto de manera superficial que es Machine Learning y por qué se debería de utilizar frente a problemas complejos, sin embargo, utilizar esta técnica trae consigo algunos retos importantes.

>*Los límites de mi lenguaje son los límites de mi mundo, los problemas técnicos se resuelven con mayor facilidad al utilizar tus herramientas conceptuales; reconoces lo que está frente a ti? sabes clasificarlo? saque que tipo de sistema de machine learning es? o por qué falla el algortimo?*

## Principales retos : E  - T - P     

### El modelo aprende de la data (E)


El rol que juega la Data es el de establecer el <a>Tipo de modelo </a> a emplear, encontrandonos con diferentes <a>tipos de data</a> con los que se puede ir trabajando, donde cada una puede acarrear un problema en particular, tenienndo soluciones diferentes para cada caso : información incompleta, valores fuera del promedio, tipos de datos distintos, etc.

Como hemos visto anteriormente, los algoritmos utilizados en los modelos recogen experiencia **E** de la información que se les pase, y en base a esta experiencia se realiza las predicciones y el reconocimiento de patrones del problema a resolver **T** , por lo que una data insuficiente, de baja calidad o que en caso no sea representativa, puede mal interpretar los datos y peor aún darnos resultados lejanos a la realidad que esperamos.

Cada estado de Data tiene un método diferente para trabajarlo, es por eso que se requiere una visualización previa mediante gráficas para poder saber que métodos usar. Por ejemplo las gráficas nos pueden decir que atributos de nuestra data se están saliendo de lo normal que representan las demás **outliers**.



### Problemas a resolver (T)

El programa recogerá la experiencia **E** de acuerdo al problema que se quiere resolver **T**, este problema es diferente índole, en el ejemplo de spam, el principal problema era reconocer que correo era un spam propiamente, entonces depende más de que se quiera solucionar con una data. Si por ejemplo se quisiera hacer un sistema para poder reconocer anomalías y prevenir el fraude electronico, el problema vendría a ser, reconocer que transacciones son los fraudes **T**


### ¿Cómo sé si mi modelo sirve? (P)

Como mencioné previamente La data para poder trabajarse en un modelo se divide en dos subconjuntos: *training set* y *test set* ambas pertenecen al mismo conjunto de datos, pero con objetivos diferentes. Sigamos enriqueciendo más nuestro enunciando principal:
Un programa aprende de cierta manera **E** con respecto al problema que se necesita resolver **T**, bajo una medida de rendimiento **P** en la resolución de cierto problema **T**. Esta medida de rendimiento se obtiene en la comparación de los resultados obtenidos del *training set* con el *test set*, debido a que los valores de entrenamiento sirven como su nombre dice para que el programa aprenda los patrones, y el set de pruebas, para verificar si las predicciones están cerca o lejos de los valores reales.
De esta forma se puede comparar que tan cerca o que tan lejos estan nuestrsa predicciones de nuestros valores, para esto existen <a>tipos de metricas</a> que nos facilitarán esta tarea.

### Pipeline 

<div style="text-align: justify;">

Luego de ver los múltiples problemas a los que se enfrenta al desarrollar un modelo, es momento de tocar un concepto que se utiliza ampliamnete, y es el de **Pipeline**. Siendo una serie de etapas donde convergen multiples modelos por medio de **Data Stores**

![Landscape](/machineretrospectiva/pipeline.png)

<figcaption>Pipeline en machine learning</figcaption>

Nuestro modelo predice cierto valor y estos valores se almacenarán en una **Data Store** para ser utilizados por otro modelo cuyo objetivo sea diferente al nuestro pero que a su vez necesite de los resultados de nuestro modelo. Cada equipo puede trabajar en un componente distinto y si algún componente fallara, los demás pueden seguir funcionando, mantiendosé una arquitectura robusta.

La siguiente imágen representa un panorama más completo de la idea de un pipeline enfocada mas al Machine Learning Operations <a>MLops</a>

![Landscape](/machineretrospectiva/pipeline2.png)

<figcaption> Vista general de un pipeline en Machine Learning - Creditos: @pawelkijko<figcaption>

</div>


## Proceso general

<div style="text-align: justify;"> 

1. Mira el panorama general (Look at the big picture)

    - Que beneficios nos dará el usar este modelo : **Objetivos del negocio**
     - Que tipo de modelo se utilizará con respecto a la data  **(E)** : <a>tipo de modelo </a>
    - Que algoritmos se van a usar **(T)** : <a>algoritmos de machine learning</a> 
    - Que tipo de métricas se usarán para evaluar el modelo **(P)** : <a>tipos de métricas</a> 
    
 
2. Obtén los datos (Get the data)

3. Descubre y visualiza los datos para obtener información (Discover and visualize the data to gain insights)

4. Prepara los datos para los algoritmos de Machine Learning (Prepare the data for Machine Learning algorithms)

5. Selecciona un modelo y entrénalo (Select a model and train it)

6. Ajusta y optimiza tu modelo (Fine-tune your model)

7. Presenta tu solución (Present your solution)

</div>



</div>

</div>


</div>