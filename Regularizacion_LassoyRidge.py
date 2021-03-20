
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error # para calcular el error medio cuadrado

from sklearn.model_selection import train_test_split #partir los datos en entranamiento y testind



def run():
    dataset = pd.read_csv('./datasets/happy/2017.csv')
    
    features = dataset[['Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.','Freedom','Trust..Government.Corruption.','Generosity','Dystopia.Residual']]
    target = dataset[['Happiness.Score']]

    x_train, x_test, y_train, y_test = train_test_split(features,target, test_size=0.25, random_state=42)

    modellinear = LinearRegression().fit(x_train,y_train)
    y_predict_linear = modellinear.predict(x_test)
    # por defecto el alpha viene dado a 1, entre mas grande sea este mayor sera la penalizacion
    modellasso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_predict_lasso = modellasso.predict(x_test)

    # ElasticNet model
    elas = ElasticNet(random_state=0)
    modelelas = elas.fit(x_train, y_train)
    y_predict_elas = modelelas.predict(x_test)

    modelrige = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = modelrige.predict(x_test)
    #para calcular la perdida usaremos el error medio cuadratico

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elas_loss = mean_squared_error(y_test, y_predict_elas)

 
    print("Linear loss:",linear_loss)
    print("Lasso loss:",lasso_loss)   
    print("Ridge loss:",ridge_loss)
    print("ElasticNet loss:",elas_loss)   
    print("::"*40)
    print("coeficientes lasso:\n",modellasso.coef_)
    print("coeficientes ridge:\n",modelrige.coef_)
    print("coeficientes ElasticNet:\n",modellinear.coef_)



if __name__ == '__main__':
    run()



#                        OUTPUT DEL ALGORITMO
#
#  Linear loss: 9.893348202868667e-08
#  Lasso loss: 0.049605751257414635
#  Ridge loss: 0.00565012452933426
#  ElasticNet loss: 1.170055134840145
#
#  Podemos observar que para este ejemplo especifico la  regularizacion ridge
#  fue la de mejor resultado ya que tiene el menor coeficiente de perdida
#
#  Coeficientes lasso:
#  [1.28921417 0.91969417 0.47686397 0.73297273 0.         0.14245522 0.89965327]
#  Coeficientes ridge:
#  [[1.07234856 0.97048582 0.85605399 0.87400159 0.68583271 0.73285696 0.96206567]]
#  Coeficientes ElasticNet:
#  [[1.00012843 0.99994621 0.99983515 1.00003428 0.99977125 1.00025981 0.99993814]]
#
#  Podemos observar que los coeficientes tiene el mismo tamaño de las columnas 
#  De los features esto es porque cada valor corresponde a cada columna de los features
#  el valor mas grande de estos coeficientes nos indican cual es la columna con mas peso
#  al momento de entrenar el modelo  y asi sucesivamente
#  podemos ver que el modelo de lazo la columna mas relevalte es la economia del pais
#  tambien vemos que este modelo elimino la columna de corrupcio "la puso en cero" porque
#  considero que este es un factor que no afecta al momento de entrenar nuestro modelo
#  vemos que en Ridge ninguno de los coeficientes es a 0 pero algunos fueron penalizados bastante 