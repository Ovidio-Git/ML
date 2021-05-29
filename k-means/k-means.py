# el hola mundo del machine learning
# es el dataset de las flores

from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

# upload dataset iris
iris = datasets.load_iris()

x_iris=iris.data
y_iris=iris.target

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

print(x.head(5))
