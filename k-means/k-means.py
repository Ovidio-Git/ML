# el hola mundo del machine learning
# es el dataset de las flores

from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def run():
    # upload dataset iris
    iris = datasets.load_iris()

    x_iris=iris.data
    y_iris=iris.target

    x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.DataFrame(iris.target, columns=['Target'])


    plt.scatter(x['Petal Length'], x['Petal Width'], c='blue')
    plt.xlabel('Petal Length', fontsize= 10)
    plt.ylabel('Petal Width', fontsize = 10)
    plt.show()
    model(x, y_iris)

def model(x, y_iris):
    # n_clusters is the same k value
    model = KMeans(n_clusters=3, max_iter=1000)
    # training
    model.fit(x)
    y_labels=model.labels_
    y_kmeans=model.predict(x)
    print("\nPredicctions: ", y_kmeans)
    accuracy = metrics.adjusted_rand_score(y_iris, y_kmeans)
    print("acurracy: ", accuracy)
    # show clusters for color
    plt.scatter(x['Petal Length'], x['Petal Width'], c=y_kmeans, s=30)
    plt.xlabel('Petal Length', fontsize= 10)
    plt.ylabel('Petal Width', fontsize = 10)
    plt.show()


if __name__ == '__main__':
    run()