#from tensorflow.keras.utils.np_utils import to_categorical
from sklearn.linear_model import LogisticRegression
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from src.models.v2.utils import *
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.models.v3.load_data import LoadData
import random
import numpy as np
#from random import randrange


# Split a dataset into k folds
from src.models.v3.utility_functions import *


def cross_validation_split(dataset, folds=3, random_state=1):
    random.seed(random_state)
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# test cross validation split



path = '../dataset/'
### Adult dataset
target_column = 'Probability'
loadData = LoadData(path)
df_adult = loadData.load_adult_data('adult.data.csv')
target_name = loadData.target_name
target_index = loadData.target_index
#print(df_adult.to_numpy())
print(df_adult.to_numpy().shape)
#data_splits = cross_validation_split(df_adult.to_numpy(), folds=10, random_state=42)
#for split in data_splits:
#    print(type(split), np.array(split).shape)

train, test = data_split(df_adult.to_numpy())
x_train, y_train = split_features_target(train, index=target_index)
x_test, y_test = split_features_target(test, index=target_index)
clf = LogisticRegression(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(y_pred)
#scores_metrics(y_pred, self.y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
count_classes = y_test.shape[1]
print(count_classes)

class Classifier:
    def __init__(self, input_shape=7, output_shape=2):
        self.model = Sequential()
        self.model.add(Dense(500, activation='relu', input_dim=input_shape))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    def fit(self, x_train, y_train, epoch=2):
        # build the model
        self.model.fit(x_train, y_train, epochs=epoch)
        return self.model
    def predict(self, x_test):
        #pred = self.model.predict(x_test)
        #y_classes = pred.argmax(axis=-1)
        #print("Class: ", y_classes)
        return self.model.predict(x_test)
    def evaluate(self, x_test, y_test, verbose=0):
        return self.model.evaluate(x_test, y_test, verbose=verbose)



model = Classifier(x_train.shape[1], count_classes)
model.fit(x_train, y_train)
pred_train = model.predict(x_train)
print(pred_train)
scores = model.evaluate(x_train, y_train)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(x_test)
scores2 = model.evaluate(x_test, y_test)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))