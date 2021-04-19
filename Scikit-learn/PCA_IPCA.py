
import pandas as pd # excel pa python
import sklearn      # scikit-learn
import matplotlib.pyplot as plt  #la de siempre bb

#importamos los modulos de sklearn que nos serviran pa la decomposicion de los features

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#para hacer prueba de rendimiento de las implemetaciones, usamos un clasificador que importamos ahora

from sklearn.linear_model import LogisticRegression

#importamos dos utilidaddes para prepara los datos antes de mandarlos a procesamiento

from sklearn.preprocessing import StandardScaler #para normalizar los datos de 0 a 1
from sklearn.model_selection import train_test_split #partir los datos en entranamiento y testind


def run():
    dt_heart = pd.read_csv('./datasets/heart/heart.csv') #cargamos el dataset
    
    dt_features = dt_heart.drop(['target'],axis=1) # sin la columa de target que es la que queremos clasificar
    #como se va hacer en las columnas ponemos acis =1
    dt_target = dt_heart['target'] 

    dt_features = StandardScaler().fit_transform(dt_features) #normalizar los features

    x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42) #partimos de los datos
    # el conjunto de entrenamiento test_size es el tamaño del conjunto de entrenamiento 
    # cada que usamos esta funcion va a partir los datos entre entrenamiento y pruebas, pero el porcentage
    # de uso de estos siempre es aleatorio , con randon state lo que hacemos es que fijamos esta aleatoriedad, para que siempre nos de 
    # la misma conclusion
    print(x_train.shape) # ver la forma de los datos .shape  output es (numeros filas, numeros columnas)
    print(y_train.shape)

    # por defecto si no ponemos un n_componentes = min (n_muestas(filas), n featues(columnas))
    pca = PCA (n_components = 3)
    pca.fit(x_train)

    #incremental pca no envia todos los datos a entrenar de una, crea pequeños bloques donde los manda a netrenar
    #poco a poco
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_) #eje y valor % que representa el dato
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(x_train)
    dt_test = pca.transform(x_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA:",logistic.score(dt_test, y_test))

    dt_train = ipca.transform(x_train)
    dt_test = ipca.transform(x_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA:",logistic.score(dt_test, y_test)) # como vemos el rendimiento es casi el mismo  y con pca solo usamos 3 features
    #y obtuvimos un resultado bueno ahorrando arto coste computacional


if __name__ == '__main__':
    run()


#                        OUTPUT DEL ALGORITMO
# (717, 13)
# (717,)
#
# SCORE PCA: 0.7857142857142857
# SCORE IPCA: 0.8051948051948052
