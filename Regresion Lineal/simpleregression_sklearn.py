
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def _train():
    dataset = pd.read_csv('./salarios_dataset.csv') # load dataser
    feature = dataset.iloc[:, :-1].values        # split feature variable
    target = dataset.iloc[:, 1].values           # split target variable
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=0) # split testing set and training set
    return x_train, y_train,x_test, y_test


def model():
    regressor = LinearRegression()
    x_train,y_train,x_test, y_test = _train()
    regressor.fit(x_train, y_train)
    # training graph
    gaph(x_train, y_train, 'g', 'r', x_train, regressor)
    # testing graph
    gaph(x_test, y_test, 'b', 'black', x_train, regressor)
    # print model score
    print(regressor.score(x_test, y_test)) # output : 0.786243  this is model accuracy 


def gaph(x, y, c1, c2, x_train, regressor):
    plt.scatter(x, y, color=c1)
    plt.plot(x_train, regressor.predict(x_train), color=c2)
    # .predict -> show prediccition information
    plt.title("Salary vs Experience")
    plt.xlabel("Feature: Experience")
    plt.ylabel("Target: Salary ")
    plt.show()


def run():
    model()


if __name__ == '__main__':
    run()