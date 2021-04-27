from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def dataset():
    dataset = load_iris()
    # print(f'Print keys of iris dataset: \n\r {dataset.keys()}') # print keys name of dataset
    # print(dataset['DESCR']) # this dataset have column named "DESC" for show data set characteristics
    # print('Species could predict: {}'.format(dataset['target_names'])) # name spacies  predict
    return dataset

def training():
    data = dataset()
    x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'],test_size =0.25,random_state=0)
    knb = KNeighborsClassifier(n_neighbors=1)
    # training model KNeighbors
    knb.fit(x_train, y_train) 
    print('Model score:',knb.score(x_test, y_test)) # output : 0.97368 this is model accuracy
    prediction = making_predictions(knb)
    print(f"Predicted name: {data['target_names'][prediction]}")

# NNOOTTEE
# scikit-learn always expects two-dimensional arrays
# for the data.

def making_predictions(model):
    # What species of iris would this be?
    x_new=np.array([[5, 2.9, 4, 0.2]])
    # To make a prediction, call the predict method:
    prediction = model.predict(x_new)
    print(f"Prediction value: {prediction}")
    return prediction
    
  

def run():
    training()

if __name__ == '__main__':
    run()




# ===================================
#   PREDICTION 1  
# 1- sepal length of  5   cm 
# 2- sepal width  of  2.9 cm
# 3- petal length of  4   cm 
# 4- petal width  of  0.2 cm

# OUTPUT 1
# Model score: 0.9736842105263158
# Prediction value: [1]
# Predicted name: ['versicolor']
# ===================================

# ===================================
# PREDICTION 2 
# 1- sepal length of  5   cm 
# 2- sepal width  of  2.9 cm
# 3- petal length of  4   cm 
# 4- petal width  of  0.2 cm

# OUTPUT 2
# Model score: 0.9736842105263158
# Prediction value: [0]
# Predicted name: ['setosa']
# ===================================