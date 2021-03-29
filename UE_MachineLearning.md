# Práctica Machine Learning (Daniel Damas)

## Módulo 1
```Python
import numpy as np
```
Se está importando la biblioteca **"NumPy"**

![](./NumPy.png) 
**Numpy** es una biblioteca para el lenguaje de programación Python que da soporte para crear *vectores y matrices grandes multidimensionales*, junto con una gran colección de funciones matemáticas de alto nivel 

```Python
import pandas as pd
```
Se está importando la biblioteca **"Pandas"**

![](./Pandas.png) **Pandas** Biblioteca que permite el *análisis de datos a través de series y «dataframes»*. Es una extensión de NumPy

```Python
import matplotlib.pyplot as plt
```
Se está importando la biblioteca **"MathPlotLib"**, especificamente la interfaz **pyplot** basada en estados.

![](./MathPlotLib.png) **Matplotlib** es una biblioteca para la *generación de gráficos* a partir de datos contenidos en listas o arrays. Se complementa con NumPy

```Python
import seaborn as sns
```
Se está importando la biblioteca **"Seaborn"**

![](./Seaborn.png) **Seaborn** es una biblioteca de *visualización de datos* de Python basada en matplotlib. Proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos e informativos

```Python
plt.style.use("ggplot")
```
Se utiliza el paquete de estilos para graficar *"ggplot"*, que está dentro de la biblioteca matplotlib.pyplot que se importó como *"plt"*

```Python
from sklearn.datasets import load_iris
```
Se importa un conjunto de datos relacionado con **orquídeas** (iris). El conjunto de datos se tomó del paper de Sir R.A.Fisher “The use of multiple measurements in taxonomic problems” Annual Eugenics, 7, Part II, 179-188 (1936); y se utiliza para reconocimiento de patrones. El conjunto de datos contiene 3 clases de 50 instancias cada una. Una clase es linealmente separable de las otras 2; estos últimos NO son linealmente separables entre sí.
Ver en: https://scikit-learn.org/dev/datasets/toy_dataset.html#iris-plants-dataset

```Python
# Cargamos las features en un DataFrame:
iris_dataset = pd.DataFrame(load_iris()["data"],columns=load_iris()["feature_names"])

# Y añadimos la columna de especies:
iris_dataset["label"] = load_iris()["target_names"][load_iris()["target"]]
```
Se crea la matriz **iris_dataset** con las columnas de interés utilizando la biblioteca *Panda*.

```Python
# Convertimos el DataFrame a jerárquico, para ser más organizados:
iris_dataset.columns = pd.MultiIndex.from_tuples(list(zip(["features"]*4, load_iris()["feature_names"])) + [("label","species")])
```
Se indexan los datos utilizando la tupla característica (4 en total) y especie.

```Python
iris_dataset.shape
```
**(150,5)**

Shape: Es la forma del conjunto de datos, significa imprimir el número total de filas o entradas **(150)** y el número total de columnas o características **(5)** de ese conjunto de datos en particular.

```Python
iris_dataset.head(10)
```
Se muestran los primeros 10 registros que se muestran a continuación.

![](./head10.png)

### Análisis del Módulo 1
En este módulo se han importado las bibliotecas matemáticas, gráficas y de visaulización de Python que se van a utilizar, se importó el set de datos, construyó la matrix e indexó, también se comprobó cuantas filas y columnas tenían, y por último, se desplegaron las 10 primeras. De momento no se pueden hacer inferencias sobre los datos.

---

## Módulo 2

```Python
iris_dataset.describe()
```
La función describe() se utiliza para generar estadísticas descriptivas que resumen la tendencia central, la dispersión y la forma de la distribución de un conjunto de datos, excluyendo los valores de NaN (del inglés "Not a Number", que significa No es un Número). 

![](./describe.png)

```Python
iris_dataset.corr()
```
La función corr() se utiliza para calcular la correlación por pares de las columnas (características o features de las flores), dependencia lineal en este caso, excluyendo los valores nulos. Los valores con mejor correlación se acercan a 1 (por eso en la diagonal hay un 1 cuando la dupla es de la misma columna), los de menor correlación se acercan a -1,  

![](./corr.png)


```Python
plt.figure(figsize=(12,6))
sns.heatmap(iris_dataset["features"].corr(), 
            vmin=-1.0,                       
            vmax=1.0,                        
            annot=True,                      
            cmap="RdBu_r")                   
                                             
pass
```
Utilizando las funciones de la biblioteca matlibplot, se define un gráfico de 12x6 pulgadas, y luego se utiliza la biblioteca seaborn para dibujar un mapa de calor con los siguientes parámetros:
 - **vmin, vmax**: Son valores mínimo y máximo para anclar el mapa de calor; de lo contrario, se infieren de los datos y otros argumentos de palabras clave. 
 - **annot**: Al ser verdadero, escribe el valor de los datos en cada celda.
 - **cmap**: Es el set de colores degradados que se muestra a continuación.

    ![](./RdBu_r.png)

Por último se ejecuta la instrucción **"pass"** que es una declaración null que se usa generalmente como marcador de posición.

El gráfico resultante es el siguiente:

 ![](./heatmap.png)

### Análisis del Módulo 2
En este módulo ya se comienza a ver información sobre el set de datos, con describe() conseguimos las valoraciones estadísticas básicas, la correlación entre las distintas características de las orquídeas (iris), y un mapa de calor donde podemos "visualizar" la información de esta correlación de una forma gráfica y con colores que según la intensidad se puede ver el nivel de correlación.

---

## Módulo 3

```Python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(iris_dataset["label","species"])
iris_dataset["label","species"] = label_encoder.transform(iris_dataset["label","species"])
iris_dataset.head()
```
El paquete sklearn.preprocessing proporciona varias funciones de utilidad comunes y clases de transformadores para cambiar los vectores de características sin procesar en una representación más adecuada para los estimadores posteriores. (Ver: https://sklearn.org/modules/preprocessing.html#preprocessing)

En este segmento se reemplaza el valor se la columna label en *"iris_dataset"*, de un valor cualitativo alfanumérico (Ej. *setosa*) a uno cuantitativo o numérico según el label_encoder. Por último se muestran las primeras 5 líneas con el comando head() como se muestra a continuación.

 ![](./head5.png)

```Python
from sklearn.model_selection import train_test_split

train, test = train_test_split(iris_dataset, train_size=0.75, test_size=0.25)
```
Con la función train_test_split de modelo de selección, se separa **Iris_Dataset** en dos sets de datos de manera aleatoria, *"train"* con un 75% de la información y *"test"* con un 25%.

```Python
train.shape
```
**(112, 5)**

Esta es la cantidad de registros y columnas de train

```Python
test.shape
```
**(38, 5)**

Esta es la cantidad de registros y columnas de test

```Python
from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier()
```
Se crea un objeto *"arbol"* que contiene las funciones que se usarán para el análisis por arbol de decisión.

```Python
grid_arbol = {"max_depth":list(range(1,11))}
```
Esto especifica la profundidad máxima a la que se construirá cada árbol, mínimo 1, máximo 11.

```Python
from sklearn.model_selection import GridSearchCV

gs_arbol = GridSearchCV(arbol,
                        grid_arbol,
                        cv=10,
                        scoring="accuracy",
                        verbose=1,
                        n_jobs=-1)
```
Se asignan los parámetros al modelo predictivo (ver: https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV.set_params)

GridSearchCV implementa un método predictivo de *"ajuste" (fit)* y *"puntuación" (Score)*.


```Python
gs_arbol.fit(train["features"], train["label","species"])
```
Se ejecuta el modelo predictivo **fit** o **ajuste** con los datos de entrenamiento (**train*). El método de **ajuste**, estima la función mas representativa para los puntos de datos suministrados (podría ser una línea, un polinomio, entre otros).

Como salida se consigue lo siguiente:

```Python
Fitting 10 folds for each of 10 candidates, totalling 100 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
[Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed:    2.3s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    2.3s finished
```
De la salida solo puedo interpretar que se han conseguido 10 cantidatos para un total de 100 pruebas que ha ejecutado y que le ha tomado 2.3 segundos ejecutando 16 workers o procesos en paralelo. Importante, esto se ha hecho con el subconjunto de entrenamiento, en el siguiente módulo se utilizará el subconjunto de test para las predicciones.

```Python
GridSearchCV(cv=10, estimator=DecisionTreeClassifier(), n_jobs=-1,
             param_grid={'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
             scoring='accuracy', verbose=1)
```
El notebook marca esta como parte de la salida anterior en Out[63], pero no debería ser una salida, lamentablemente no tengo posibilidad de instalar Python en mi equipo para probarlo. Estos parámetros son casi idénticos a los asignados a **gs_arbol** en el paso In[63], salvo la profundidad que antes era de 1..11 y ahora está de 1..10.

### Análisis del Módulo 3
En este módulo basicamente se dividen los datos en un conjunto de entrenamiento (*train 75%*) y uno de pruebas (*test 25%*), de esta manera se establecen los parámetros del modelo utilizando arboles de decisión y luego se entrena utilizando ajuste (*fit*) de precisión (*accuracy*).


---

## Módulo 4

```Python
from sklearn.metrics import accuracy_score

accuracy_en_test = accuracy_score(y_true = test["label","species"],
                                  y_pred = gs_arbol.best_estimator_.predict(test["features"])
                                 )

print("%s" % accuracy_en_test)
```
Puntuación de precisión: esta función calcula la *precisión del subconjunto*, en este caso se está utilizando el subconjunto de **test**, el conjunto de etiquetas predichas para una muestra debe coincidir exactamente con el conjunto de etiquetas correspondiente a las **especies**.

En este caso se consigue el siguiente resultado.

**0.9473684210526315**

Esto indica que la precisión del modelo es del 94.7%

```Python
from sklearn.metrics import confusion_matrix

matriz_confusion = confusion_matrix(y_true = test["label","species"],
                                    y_pred = gs_arbol.best_estimator_.predict(test["features"])
                                   )

matriz_confusion
```
Se calcula la matriz de confusión para evaluar la precisión de la clasificación de las especies en **test**. Por definición, una matriz de confusión es igual al número de observaciones conocidas que están en grupo *y_true* y que se predice que están en grupo *y_pred*. En este caso se está utilizando para *y_pred* el arbol de decisión que se preparó con los datos de *train* pero en este caso el modelo utilizará las características del conjunto de datos de **test** como los valores conocidos o verdad.

Se imprime la matriz resultante.

```Python
array(
    [[12,  0,  0],

     [ 0, 11,  1],

     [ 0,  1, 13]])
```

En este caso se consiguen solo dos (2) *"no coincidencias"* para los 38 elementos de la muestra de **test**, que coincide con el calculo de la precición anterior del 94.7%.

### Análisis del Módulo 4
En este módulo se estima la precisión del **conjunto de datos de prueba (test)**, siendo esta de un 94,7% y luego se utiliza una matriz de confución para validar donde están las valores falsos no no coincidentes. 

Como conclusión final, dado un conjunto de características (4) del objeto en estudio, el modelo puede predecir con una fiabilidad alta de aprox 95%, el tipo de especie de *orquídea (iris)*, según vimos en clase, mientras mas amplio y de calidad sean los datos de entrenamiento, mayor será la precisión del modelo, en particular. Me ha quedado claro lo que se puede hacer con librerías o bibliotecas matemáticas y estadisticas, con los modelos predictivos y las distintas formas de presentar la información, así como ver las correlaciones entre los datos para poder tomar decisiones con base en los resultados.


---

## Enlaces consultados

https://www.w3resource.com/pandas/dataframe/dataframe-describe.php

https://scikit-learn.org/dev/datasets/toy_dataset.html#iris-plants-dataset

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure

https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap

https://gallantlab.github.io/colormaps.html

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_tuples.html
