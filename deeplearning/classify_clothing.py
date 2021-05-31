from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def model(train_images, train_labels, test_images, test_labels):
    model=keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
    # how this process is long we need optimizar with
    model.compile(tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # trainig model, epochs -> interactions number for trainig
    model.fit(train_images, train_labels, epochs = 5)
    # if the epochs is more larger the acurracy also
    # with a epoch = 5 the accuracy is = 0.84
    # with a epoch = 8 the accuracy is = 0.88
    # but a intactions  up to accuracy: 0.9043
    # evaluate model accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    predictions = model.predict(test_images)
    graph(predictions=predictions, ctrl=3, test_images=test_images, test_labels=test_labels)


def graph(train_images=None, train_labels=None, predictions=None, test_images=None,test_labels=None, ctrl=None):
    # define tags
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    if ctrl == 1:
        # show image number 100
        plt.figure()
        plt.imshow(train_images[100])
        plt.grid(True)
    elif ctrl == 2:
        #show dataset
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
    elif ctrl == 3:
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            plt.imshow(test_images[i], cmap=plt.cm.binary)
            predicted = np.argmax(predictions[i])
            true_label = test_labels[i]
        if predicted == true_label:
            color='green'
        else:
            color='red'
        plt.xlabel('{} ({})'.format(class_names[predicted], class_names[true_label], color=color))


def run():
    # upload  dataset of clothing image
    # this dataset have 60.000 images
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # slicing sets
    train_images=train_images/255.0
    test_images=test_images/255.0
    model(train_images, train_labels, test_images, test_labels)


if __name__=='__main__':
    run()
