import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import tensorflow as tf
import random
from sklearn.neighbors import KNeighborsClassifier

SEED = 100
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def read_data(path):
    data = pd.read_csv(path, sep=' ')

    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    for i in range(0, len(labels)):
        if labels[i] == 1:
            labels[i] = 0
        if labels[i] == 2:
            labels[i] = 1
        if labels[i] == 3:
            labels[i] = 2

    return features, labels


def create_model(features, labels):
    # Create the Sequential model
    model = Sequential()
    model.add(Dense(11, activation='softmax', input_dim=13))
    # model.add(Dense(32, activation='softmax'))
    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=0.3, momentum=0.5, nesterov=True, decay=0.01)
    adam = Adam()  #
    rms = RMSprop(lr=0.003)  #

    model.compile(adam, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(features, labels, epochs=1000, verbose=2, batch_size=100)

    return model


def knnClassifer(trainingDataPath,testDataPath):
    print('-----KNN ------')
    knn = KNeighborsClassifier(n_neighbors=7)
    features, labels = read_data(trainingDataPath)
    features2, labels2 = read_data(testDataPath)

    knn.fit(features, labels)
    y_predict = knn.predict(features2)
    print('-----predict value is ------')
    print(y_predict)
    print('-----actual value is -------')
    print(labels2)
    count = 0
    for i in range(len(y_predict)):
        if y_predict[i] == labels2[i]:
            count += 1
    print('accuracy is %0.2f%%' % (100 * count / len(y_predict)))


if __name__ == '__main__':
    # # # # # # # # # # # # # # Please change path here# # # # # # # # # # # # #
    features_train, labels_train = read_data("/Users/Patrick/Desktop/COMP307/A2/ass2_data/part2/wine_training")
    Model = create_model(features_train, labels_train)
    # # # # # # # # # # # # # # Please change path here# # # # # # # # # # # # #
    features_test, labels_test = read_data("/Users/Patrick/Desktop/COMP307/A2/ass2_data/part2/wine_test")

    score_train = Model.evaluate(features_train, labels_train)
    print("\nAccuracy: ", score_train[-1])

    # Checking the predictions
    print("\nPredictions:")
    print(Model.predict(features_train))

    score = Model.evaluate(features_test, labels_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # knnClassifer("/Users/Patrick/Desktop/COMP307/A2/ass2_data/part2/wine_training","/Users/Patrick/Desktop/COMP307/A2/ass2_data/part2/wine_test")
