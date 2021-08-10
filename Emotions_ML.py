# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Importing Data
emotions = pd.read_csv("C:/Users/jeanp/OneDrive/GitHub/Emotions Karas Deep Learning/Data/emotions.csv")
emotions.info()


#Obtaining EGG features
features = emotions.loc[0:len(emotions), 'fft_0_b':'fft_749_b']

encoder = LabelEncoder()
encoder.fit(emotions["label"])
response = encoder.transform(emotions["label"])


#Train test Split
x_train, x_test, y_train, y_test = train_test_split(features, response, test_size=0.2)
y_train = to_categorical(y_train, num_classes = None)


#Creating Neural Network
n_col = features.shape[1]
model = Sequential()
model.add(Dense(32, activation = "relu", input_shape = (n_col, )))
model.add(Dense(32, activation = "relu"))
model.add(Dense(3, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train)
print(model.summary)


model_acc = model.evaluate(X_test,y_test,verbose=0)