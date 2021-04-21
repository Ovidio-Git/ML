
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns



def _train():
    dataset = pd.read_csv("./diabetes_dataset.csv")
    feature_col = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
    features = dataset[feature_col]
    target = dataset.Outcome
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state= 0)
    return x_train, x_test, y_train, y_test


def model_logistic():
    x_train, x_test, y_train, y_test = _train()
    logistique = LogisticRegression()
    logistique.fit(x_train, y_train) # training start
    y_pred = logistique.predict(x_test)
    return y_pred, y_test


def matriz_confussion():
    y_pred, y_test = model_logistic()
    matrix = metrics.confusion_matrix(y_test, y_pred)
    # graph matrix 
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick = np.arange(len(class_names))
    plt.xticks(tick, class_names)
    plt.yticks(tick, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap='Blues_r', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confussion matrix', y = 1.1)
    plt.ylabel('Actual')
    plt.xlabel('Prediccion')
    print("exactitud:", metrics.accuracy_score(y_test, y_pred))

def run():
    matriz_confussion()


if __name__ == '__main__':
    run()


# en la matrix de confusion el valor en blanco muesra los valores correctos
# la diagonal hacia a abajo muestra los que son negativos seguros
# el que esta arriba de la diagolnal negativa son los que son clasificados incorrectamente
# como negativos, osea que enrealidad fueron positivos pero salen en negativos
# el diagonal de este nos muestra los positivos que no devieron ser clasificados
# de esta forma 