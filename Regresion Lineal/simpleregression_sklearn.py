
import pandas as pd 
import numpy as np 
import random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def _countrys():
    country = { 1:'COL',
                2:'MEX',
                3:'ARG',
                4:'RUS'}
    contry = np.array([[rd.choice([1,2,3,4]) for i in range(30)]])
    return contry

def model_salary_vs_countrys():
    dataset = pd.read_csv('../datasets/salarios_dataset.csv') # load dataser
    feature = np.reshape(_countrys(), (30,1)) # converting list 1D to 2D
    target = dataset['Salario'].values           # split target variable
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=0) # split testing set and training set
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    gaph(x_train, y_train, 'r', 'y', x_train, regressor)
    print(regressor.score(x_test, y_test)) # output : 0.786243  this is model accuracy 


def _train():
    dataset = pd.read_csv('../datasets/salarios_dataset.csv') # load dataset
    feature = dataset.iloc[:, :-1].values           # split feature variable
    target = dataset.iloc[:, 1].values              # split target variable
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=0) # split testing set and training set
    return x_train, y_train,x_test, y_test


def model_salary_vs_years():
    regressor = LinearRegression()
    x_train,y_train,x_test, y_test = _train()
    regressor.fit(x_train, y_train)
    # training graph
    gaph(x_train, y_train, 'g', 'r', x_train, regressor)
    # testing graph
    gaph(x_test, y_test, 'b', 'black', x_train, regressor)
    # print model score
    print(regressor.score(x_test, y_test)) # output : 0.786243  this is model accuracy 
    making_predictions(regressor)

def making_predictions(model):
    # What species of iris would this be?
    x_new=np.array([[2]])
    # To make a prediction, call the predict method:
    prediction = model.predict(x_new)
    print(f"Prediction value: {prediction}")

    

def gaph(x, y, c1, c2, x_train, regressor):
    plt.scatter(x, y, color=c1)
    plt.plot(x_train, regressor.predict(x_train), color=c2)
    # .predict -> show prediccition information
    plt.title("Salary vs Experience")
    plt.xlabel("Feature: Experience")
    plt.ylabel("Target: Salary ")
    plt.show()


def run():
    model_salary_vs_years()
    #model_salary_vs_countrys()


if __name__ == '__main__':
    run()