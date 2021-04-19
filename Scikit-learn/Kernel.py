
import pandas as pd # excel pa python
import sklearn      # scikit-learn
import matplotlib.pyplot as plt  #la de siempre bb

#importamos los modulos de sklearn que nos serviran pa la decomposicion de los features

from sklearn.decomposition import KernelPCA

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

    # el cambio de esta funcion con los otros pca es el parametro kernel 
    # para definir los tipos de kernel 'linear(este es equivalente a trabajar un pca normal)' 'poly'
    # 'rbf'-> kernel gausiano tipo rbf 
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(x_train)

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("SCORE KPCA:", logistic.score(dt_test, y_test))
    

    
if __name__ == '__main__':
    run()



#                        OUTPUT DEL ALGORITMO
#
#  SCORE KPCA: 0.7987012987012987
#