

import pandas as pd 
from sklearn .linear_model import (  #Libreria de skitlearn para trabajar datos atipicos
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





def run():
    dt_happy = pd.read_csv('./datasets/happy/2017.csv') 
    

    dt_features = dt_happy.drop(['Country','Happiness.Score'], axis=1)
    dt_target = dt_happy[['Happiness.Score']]
    x_train, x_test, y_train, y_test= train_test_split(dt_features,dt_target, test_size=0.3, random_state=42)

    # forma profesional de manejar estimadores
    estimadores = {
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER' : HuberRegressor(epsilon=1.35)
    }

    for key, estimador in estimadores.items():
        estimador.fit(x_train, y_train)
        predictions = estimador.predict(x_test) 
        
        print("=="*30)
        print(key)
        print("MSE: ",mean_squared_error(y_test, predictions))

if __name__ == '__main__':
    run()



#                             OUTPUT DEL ALGORITMO 
# ============================================================
# SVR
# MSE:  0.017468243325813742
# ============================================================
# RANSAC
# MSE:  2.505044239131515e-29
# ============================================================
# HUBER
# MSE:  3.4447501305057568e-06
# ============================================================

# Cabe recalcar que este no es el outpu literal, ya que al correr este codigo
# salen impresos algunos warnings pero no afectan su funcionamiento.

# recordemos que MSE es el error cuadratico medio de cada estimador, recordemos
# que este concepto hace referencia  a la diferencia entre el estimador y lo que 
# se estima 

# al dejar esto en claro y analizar los resultados podemos encontrar que con el estimador
# SVR es mucho mayor el porcentaje el error cuadratico medio debido a que introducimos algunos
# datos corruptos al dataset 

# pero al realizar la regresion robusta con los estimadores tanto RANSAC como HUBER este error
# se disminuyo desticamente debido a la accion de estos algoritmos ya que lo que hacen es reducir la influencia
# de los valores atipicos a la medicion 