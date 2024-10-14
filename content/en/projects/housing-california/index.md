---
title: Housing Price Predictor
date: 2024-10-04
author: David Vasquez
draf : false
description: Predecir el precio de las viviendas en el estado de california
tags: Machine Learning, Prediction, houses, values, pipelines
---

<div style="text-align: justify;">

## Introducción

### Objetivo

Lograr predecir el precio de las viviendas en el estado de California utilizando técnicas de Machine Learning validando todos

 nuestros resultados con técnidas de validación que nos permitan optimizar nuestro modelo, obteniendo los mejores resultados. 
 Utilizando la métodología presentada en <a href="/posts/prologo/" target="_blank">Machine learning, retrospectiva </a> que se pueda aplicar a cualquier otro distrito bajo la misma metodología.


### Descripccion

Aprovechar todas las dimensiones del dataset <a href="https://www.kaggle.com/datasets/camnugent/california-housing-prices" target="_blank">California Housing Prices</a> para poder hacer un análisis clasificado como **analítica descriptiva** de cuales son las variables que se relacionan y en base a eso aplicar métodos de machine learning 
que nos permitan cumplir con nuestro objetivo como: <a href="https://www.ibm.com/mx-es/topics/random-forest" target="_blank">Random Forest Regresor</a>, <a>GridSearchCv </a> entre otras. 

## Desarrollo

### Mira el panorama general 

Lo primero que debemos de hacer es ver todas las variables que tienen nuestra dataset que para este ejemplo tendrá el noombre de **housing**, ya que estas estarán involucradas en la predicción de nuestra variable objetivo que es el precio de la vivienda, y en la selección del tipo de sistema que vamos a emplear, en este caso <a>supervisado</a>; ya que todas las columnas tienen sus respectivos valores, o en términos técnicos, cada label tiene su output.




En <a href="/posts/prologo/" target="_blank">Machine Learning, retrospectiva </a> aprendimos que los principales retos a resolver en Machine Learning aplicados a cualquier proyecto:

- **E** Que aprenderá de la data? Como hemos visto es un modelo supervisados
- **T** Como se resolverá este problema: Se quiere predecir el valor de una vivienda, el algoritmo acorde es **regresión** ya que predice los valores de una variable. Pero no es el único, por lo que se recomienda investigar que tipo de algoritmos pertenecen a cada tipo de sistema, teniendo todas las opcciones a la mano y el diferente uso.
- **P** Que metricas usaremos ? : Esto lo veremos mas adelante.


## EDA

Comenzamos con el Analisis exploratorio de Datos utilizando herramientas que nos proporcionará la herramientas **Pandas**, este un proceso que se utiliza antes de crear el modelo. Permitiendonos observar como está estructurada nuestra data y cuales serian las medidas que tenemos que tomar antes de empezar a trabajar nuestro modelo, obteniendo mejores resultados en las predicciones.


<figcaption>housing.head()</figcaption>

![Landscape](/housing-california/heads.png)

La primera imagen en la que se aplica el comando head() nos devolvéra las cinco primeras filas, con sus valores y el nombre de cada columna. Lo que podemos aprender de esta imagen es al tipo de valor que nos vamos a enfrentar, por ejemplo vemos **longuitud, latitud** que pueden utiles para la representación en un **mapa**, 
y al final observamos una columna con el nombre **ocean_proximiy**, y los valores son de tipo categórico. Este tipo de valor nos adelanta que debemos de
 transformarlo a número bajo el contexto de la data y las predicciones que queremos usar, este tipo de transformación lleva el nombre de
  <a href="https://www.datacamp.com/es/tutorial/one-hot-encoding-python-tutorial" target="_blank">One hot encoding</a>, que involucra sus propias dificultades y transformaciones al trabajar con este tipo de valores.



<figcaption>housing.info()</figcaption>

![Landscape](/housing-california/infos.png)

Aplicando el siguiente comando nos ofrece la siguiente información
El segundo comando nos permite saber cuantas lineas tiene nuestro datase :
-   **RangeIndex** : Los valores de 20640 entradas, desde e 0 hasta el 20639; el total de filas es de 2060 filas.
-  **Data Columns**: El número total de columnas, 10 columnas.
-  **Column**: Nombres de las columnas con sus respectivos valores no nulos, y el tipo de dato.
-  **Non-Null Count**: El numero de datos que no estan vacios. La columna llamada *Total_bedrooms* presenta valores que estan vacios, lo cual es importante tenerlo en cuenta para mas adelante.
-  **Dtype** : Nos dice el tipo de dato que contiene cada columna. Nueve del tipo flotante(decimal) y uno de tipo objeto(texto o categoría).

### Identificar patrones: Observar las tendencias numericas y visuales que nos ayudarán a construir nuestro modelo

<figcaption>housing.describe()</figcaption>

![Landscape](/housing-california/describe.png)

- **Limitadores (capped)** : Como podemos observar las barras enmarcadas en el rectangulo rojo escapan de la distribución normal de datos, en *housing_media_age* y en *media_house_value*, esto se conoce como **cappesd**, significa que hay casas mayores a 50 años de antiguedad que se han agrupado en el mismo valor, y las casas con un valor mayor a $500, 000 fueron agrupadas a ese valor.


- Detectar los outliers data: Encontrar cualquier punto de la data que pueda afectar la calidad del modelo. Un outlier es un caso especial que no representa la mayoría de la data. Estos valorse deben de ser eliminados para evitar que afecten las predicciones ya que lo sesgarían. Generando distorsión.

- Generar una hipotises inicial: Obtener algunas ideas del comportamiento de la data para ser testeadas luego con el proceso de modelado, por ejemplo, a golpe de vista con los graficos podeemos creer que la varible que mas se relaciona es la del area:


## Creando un conjunto de entrenamiento y de validación

Se separa una cantidad de datos para el test de entrenamiento, por convención : traning_set , y otro test para  validación , por convención test-set,
con el fin de evitar cualquier sesgo mental que influya en la decisión de nuestro modelo o algoritmo - overfitting menta. Esta separación la realizamos para evitar también el **data snooping bias** el cúal es un efecto que Ocurre cuando exploras el conjunto de datos de prueba y en base a esto tomas decisiones sobre tu modelo, creamos por lo tanto dos sets, un *test_Set* y un *train_set*.

1. Nuestro dataset **housing** contiene un variable llamada income_cat, la cúal fue convertida a sus valores categóricos.

    <div style="text-align: center;">

    <figcaption>Estratificación<figcaption>

    ![Landscape](/housing-california/strat.png)

    </div>

    La función **split.split(housing, housing["income_cat"])** , devuelve una tupla con dos listas de índices : train_index y test_index, y el bucle, almacena ambas listas en variables que tienen el mismo nombre.
    De esta forma ya tenemos separadas las filas que irán a nuestro set de entrenamiento y prueba. llamadas **strat_train_set** y **strat_test_set** que están proporcionalmente distrubidas con los valores de income_cat; para valores de entrenamiento y prueba

    <div style="text-align: center;">

    <figcaption>Separar nuestra variable objetivo<figcaption>

    ![Landscape](/housing-california/predi.png)

    </div>

    Una vez hecho esto, vamos a guardar todos los valores de entrenamiento en un variable **housing** , pero sin nuestra variable objetivo, ya que está ira en otra variable aparte, para que sirve como variable predictora.




### Corrigiendo valores nuestros sets

Gracias a nuestro análisis exploratorio de datos (EDA) del comienzo pudimos percatarnos de las particularidades de nuestra data, entre ellas los valores faltantantes en la columna **total_bedrooms** y el tipo de dato categórico en la **ocean_proximiy**; es momento de aplicarles las transformaciones necesarias antes de separar nuestros sets.
1.  Comenzemos con **total_bedrooms**:
    Se pueden llenar los datos de tres formas:
     - Reemplazar los valores con 0 : Puede traer problemas en la predicción la ausencia de valores
     - Reemplazar los valores con el promedio : Si hay outliers dentro de nuestra data, esto puede distorsionar los resultados
     - Reemplazar los valores con la mediana (valor central) : Puede no ser tan útil si tu data es simétrica.

    El método imputer nos permite aplicar la mediana a todos los valores numéricos
    
    <figcaption>El método imputer nos ofrece la estrategia mediana<figcaption>

    ![Landscape](/housing-california/mii.png)


2. Nuestra data es simétrica, sin embargo en pro de utilizar las mejores prácticas en mira a encontrarnos con los peores escenarios, utilizaremos la mediana para llenar los valores faltantes y excluiremos los valores de la columna categórica, porque la mediana solo trabaja con valor numéricos, colocamos  **ocean_proximity** en otra variable:


    Aplicando<a> One Hot Enconding</a> a nuestros valores categóricos.

    <div style="text-align: center;">

    <figcaption>One-Hot-Encoding<figcaption>

    ![Landscape](/housing-california/1hot.png)

    </div>


    Transformar los datos con esta técnica nos permite que no haya relación de orden entre los valores de estos datos, debido a que para el programa se puede volver ambiguo si las 
     transformaciones son por **ordinal encoder** que asigna un número a cada categoría, habiendo errores de falsa proximidad.
    Para una explicación mas completa te recomiendo leer
     <a href="https://www.datacamp.com/es/tutorial/one-hot-encoding-python-tutorial" target="_blank">one-hot-enconding</a> donde ahondo mucho mas en esta ventaja.


## Transformadores

Una vez que nuestros sets esten preparados de la mejor manera es momento de aplicar transformadores, estos nos permiten modificar todos los valores tanto de las filas como de las columnas al mismo tiempo, aplicando las
transoformaciones necesarias a cada valor, sin necesidad de tener que hacerlo individualmente para cada uno.
      
### PipeLine y Full PipeLine

1. Pipeline es el flujo de trabajo con el que se irá aplicando una transformación específica a cada elemento de nuestros sets, previamente configurada o por defecto a cada variable dentro de nuestors parámetros, permtiendonos hacer transformaciones en orden; el orden es importante


    <div style="text-align: center;">

    <figcaption>Pipeline para numeros y luego full pipeline<figcaption>

    ![Landscape](/housing-california/pip.png)

    </div>


    Podemos observar que lo aplicado anteriormente converge dentro de una misma función y que ambas son depedientes, por ejemplo empezando por **full pipeline**, depende de **pipeline**. El transformador que se configuró previamente **num_pipeline** es usado como parámetro para ser aplicado a cada elemento, de **num_atribs**  y que **OneHotEnconder** también es utilizado, y , aunque este pertenece a un proceso antes, podemos saber lo que hará en este caso y el valor que nos devolverá.



## Selecciona un modelo y entrénalo 

Luego de analziar los modelos como <a>linear regresion</a> , <a>arbol de decision</a> y <a>Random Forest Regresor</a>, nos quedaremos con el modelo que tenga una menor desviación estandar y nuestrá de métrica de errr también sea la menor.


1. Valores encontrados por modelo:
   - Linear regresion : 
     - Mean: 69104.07998247063
     - Standard deviation: 2880.328209818069.
   - Decision Three : 
     - Mean: 71183.31910562774
     - Standard deviation: 3183.5691966642676.
   - Random Forest :
     - Mean: 50378.11605323106
     - Standard deviation: 2211.9077934533375

Según los valores presentados , el modelo que mejor scores presenta es el de **Random Forest**

## Ajusta y optimiza tu modelo 

GridSearchCV es crucial en el proceso de optimización de modelos de machine learning, ya que permite encontrar la mejor combinación de hiperparámetros para un algoritmo específico de manera sistemática y eficiente. Al explorar exhaustivamente un espacio de parámetros predefinido, GridSearchCV evalúa el rendimiento del modelo utilizando técnicas de validación cruzada, lo que ayuda a evitar el sobreajuste y a garantizar que el modelo generalice bien en datos no vistos.GridSearchCV ajustará todos los parámetros de nuestro random forest con la validación cruzada

<div style="text-align: center;">

<figcaption>El modelo se encuentra ajustado<figcaption>

![Landscape](/housing-california/solucion.png)

</div>

## Presenta tu solución 

Nuestro modelo ahora que funciona correctamente al presentar una distribución lineal con poca perturbación ya está habilitado para que se utilice en la predicción de viviendas de valor actual. Puedes encontrar la aplicación en el siguiente enlace. <a href="https://housing-price-predictor-lazojdavid.streamlit.app/" target="_blank">Web App Housing Price Predictor</a>






</div>








</div>
