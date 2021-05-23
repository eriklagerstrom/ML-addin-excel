import xlwings as xw
import numpy as np
import pandas as pd
from sklearn import *
import sklearn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import initializers

""" boston = datasets.load_boston()

data = boston.target

print(data[0])

targets = pd.DataFrame(boston.target)
features = pd.DataFrame(boston.data)

features.columns = boston.feature_names

targets.to_excel("boston_targets.xlsx")
features.to_excel("boston_features.xlsx")

print(features.describe()) """


def scale(X, XForecast):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler.fit(X)
    X = scaler.transform(X)
    XForecast = scaler.transform(XForecast)
    
    return X, XForecast

hiddenLayerNodes = [32, 32]
classification = False
activation = "relu"
epochs = 5
batchSize = 3
X = np.array(pd.read_excel("boston_x_train.xlsx"))
Y = np.array(pd.read_excel("boston_y_train.xlsx"))
XForecast = np.array(pd.read_excel("boston_x_forecast.xlsx"))

X, XForecast = scale(X, XForecast)

if classification:
    metric = 'accuracy'
    if Y.shape[1]>1:
        # ASSUMES labels are one hot encoded
        output_activation = "softmax"
        loss = "categorical_crossentropy"
    else:
        # ASSUMES labels of 0 or 1
        output_activation = "sigmoid"
        loss = 'binary_crossentropy'
else:
    output_activation = "linear"
    metric = "mae"
    loss = "mse"
    
model = Sequential()

# First layer added separately due to input_dim argument
model.add(Dense(hiddenLayerNodes[0], input_dim = X.shape[1], kernel_initializer = initializers.he_uniform, activation = activation))

# Only loop to add middle layers except first
for i in range(1, len(hiddenLayerNodes)):
    model.add(Dense(hiddenLayerNodes[i], activation = activation, kernel_initializer = initializers.he_uniform))
    
# Add last layer separately due to output activation function
model.add(Dense(Y.shape[1], activation = activation, kernel_initializer = initializers.he_uniform))

model.compile(loss=loss, optimizer = "adam", metrics = metric)
model.fit(X, Y, epochs = epochs, batch_size = batchSize, verbose = 0)

predicted = model.predict(XForecast)

df = pd.DataFrame(predicted)
df.to_excel("predicted_test.xlsx")
