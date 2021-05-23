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

'''

Gör SVC/SVR
MLP
RNN
LSTM
CNN
XDGBoost

'''

def scale(X, XForecast):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler.fit(X)
    X = scaler.transform(X)
    XForecast = scaler.transform(XForecast)
    
    return X, XForecast


def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"


@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array)
@xw.arg("XForecast", np.array, ndim = 2)

def linreg(X, Y, XForecast, classification):

    if len(X.shape) == 1:
        X = X.reshape(-1,1)
        XForecast = XForecast.reshape(-1,1)
   
    X, XForecast = scale(X, XForecast)
    if classification:
        model = sklearn.linear_model.LogisticRegression(random_state = 0)
        model.fit(X, Y)
        
    else:
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, Y)
    
    predicted = model.predict(XForecast)
    
    return predicted
    
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def nn_simple(X, Y, XForecast, hiddenLayerNodes, classification, activation, epochs, batchSize):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float) 
    X, XForecast = scale(X, XForecast)

    df_tmp = pd.DataFrame(XForecast)
    df_tmp.to_excel("XForecast.xlsx")

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
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializers.he_uniform))
    
    model.compile(loss=loss, optimizer = "adam", metrics = metric)
    model.fit(X, Y, epochs = epochs, batch_size = batchSize, verbose = 1)
    
    predicted = model.predict(XForecast)

    return predicted
    
if __name__ == "__main__":
   xw.Book("myproject.xlsm").set_mock_caller()
   main()

