import xlwings as xw
import numpy as np
import pandas as pd
from sklearn import *
import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Dropout
from tensorflow.keras import initializers
import keras
import os

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
    
    #return sklearn.preprocessing.normalize(X, norm='l2'), sklearn.preprocessing.normalize(XForecast, norm='l2')


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
    

# VANILLA NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def nn_simple(X, Y, XForecast, hiddenLayerNodes, classification, activation, epochs, batchSize, lr):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

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
    initializer = initializers.HeUniform()
    
    # First layer added separately due to input_dim argument
    model.add(Dense(hiddenLayerNodes[0], input_dim = X.shape[1], kernel_initializer = initializer, activation = activation))
    
    # Only loop to add middle layers except first
    for i in range(1, len(hiddenLayerNodes)):
        model.add(Dense(hiddenLayerNodes[i], activation = activation, kernel_initializer = initializer))
        
    # Add last layer separately due to output activation function
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(X, Y, epochs = int(epochs), batch_size = int(batchSize), verbose = 1)


    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "nerual net model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(XForecast)
    return predicted


#RECURRENT NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def rnn_py(X, Y, XForecast, hiddenLayerNodes, classification, activation, epochs, batchSize, lr,
            rnnActivation, rnnType, dropout, rnnDropout, rnnLookback):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

    X, XForecast = scale(X, XForecast)
    rnnLookback = int(rnnLookback)

    dropout = float(dropout)
    rnnDropout = float(rnnDropout)
    num_features = X.shape[1]
    rows_train = X.shape[0]-rnnLookback
    rows_forecast = XForecast.shape[0]-rnnLookback

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
        
    x_reshaped = np.array([])
    y_reshaped = np.array([])
    x_forecast_reshaped = np.array([])

    # Reshape the array according to look-back period provided by the user
    for i in range(int(rnnLookback), int(X.shape[0])):
        x_reshaped = np.append(x_reshaped, X[(i-rnnLookback):i+1, :])
        y_reshaped = np.append(y_reshaped, Y[i, :])

    for i in range(int(rnnLookback), int(XForecast.shape[0])):
        x_forecast_reshaped = np.append(x_forecast_reshaped, XForecast[(i-rnnLookback):i+1, :])
    
    try:
        return_sequences = [True for i in range(len(hiddenLayerNodes))]
    except:
        hiddenLayerNodes = [hiddenLayerNodes]
        return_sequences = [True]
    
    return_sequences[-1] = False
    x_reshaped = np.reshape(x_reshaped, (rows_train, rnnLookback+1, num_features))
    x_forecast_reshaped = np.reshape(x_forecast_reshaped, (rows_forecast, rnnLookback+1, num_features))
        
    model = Sequential()
    initializer = initializers.HeUniform()

    if rnnType == "RNN":
        #, input_dim = (x_reshaped.shape[0]'
        model.add(SimpleRNN(units = hiddenLayerNodes[0], input_shape =(rnnLookback+1, num_features), kernel_initializer = initializer, activation = activation,
                    return_sequences = return_sequences[0], dropout = dropout, recurrent_dropout = rnnDropout))
        
        for i in range(1, len(hiddenLayerNodes)):
            model.add(SimpleRNN(units = hiddenLayerNodes[i],return_sequences = return_sequences[i], activation = activation, kernel_initializer = initializer,
            dropout = dropout, recurrent_dropout = rnnDropout))

    elif rnnType == "LSTM":
        #, input_dim = (x_reshaped.shape[0]
        model.add(LSTM(units = hiddenLayerNodes[0], input_shape =(rnnLookback+1, num_features), kernel_initializer = initializer, activation = activation,
            return_sequences = return_sequences[0], recurrent_activation = rnnActivation, dropout = dropout, recurrent_dropout = rnnDropout))
        
        for i in range(1, len(hiddenLayerNodes)):
            model.add(LSTM(hiddenLayerNodes[i],return_sequences = return_sequences[i], activation = activation, kernel_initializer = initializer,
             recurrent_activation = rnnActivation, dropout = dropout, recurrent_dropout = rnnDropout))
    
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(x_reshaped, y_reshaped, epochs = int(epochs), batch_size = int(batchSize), verbose = 1)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "rnn model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(x_forecast_reshaped)
    return predicted


#CONVOLUTIONAL NEURAL NETWORK
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def cnn_py(X, Y, XForecast, outputLayerNodes, classification, activation, epochs, batchSize, lr,
            dimension, filters, kernelSize, groups, strides, padding, dilationRate):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

    X, XForecast = scale(X, XForecast)
    rnnLookback = int(rnnLookback)

    num_features = X.shape[1]
    rows_train = X.shape[0]-rnnLookback
    rows_forecast = XForecast.shape[0]-rnnLookback

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
    initializer = initializers.HeUniform()
    inputShape = []

    #Different type of layer used depending on what dimension user put in
    if dimension == 1:
        model.add(Conv1D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling1D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv1D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling1D(pool_size = 2, filters = filters[i]))

    elif dimension == 2:
        model.add(Conv2D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling2D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv2D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling2D(pool_size = 2, filters = filters[i]))
    
    else:
        model.add(Conv3D(filters[0], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation, input_shape = inputShape)
        model.add(MaxPooling3D(pool_size = 2, filters = filters[0]))
        
        for i in range(1, len(filters)):
            model.add(Conv3D(filters[i], kernelSize, strides, padding, dilation_rate = dilationRate, groups = groups, activation = activation)
            model.add(MaxPooling3D(pool_size = 2, filters = filters[i]))
        
    model.add(Flatten())
    model.add(Dense(outputLayerNodes, activation = activation, kernel_initializer = initializer)
    model.add(Dense(Y.shape[1], activation = output_activation, kernel_initializer = initializer))

    opt = keras.optimizers.Adam(learning_rate = float(lr))
    model.compile(loss=loss, optimizer = opt, metrics = metric)
    history = model.fit(x_reshaped, y_reshaped, epochs = int(epochs), batch_size = int(batchSize), verbose = 1)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "rnn model log.txt"), "w")
    f.write(str(history.history))
    f.write('\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()

    predicted = model.predict(x_forecast_reshaped)
    return predicted


# SUPPORT VECTOR MACHINE
@xw.func
@xw.arg("X", np.array, ndim = 2)
@xw.arg("Y", np.array, ndim=2)
@xw.arg("XForecast", np.array, ndim = 2)
def svm_py(X, Y,XForecast, kernel, degree, C, gamma, classification):

    X = X.astype(np.float)
    XForecast = XForecast.astype(np.float)
    Y = Y.astype(np.float)

    X, XForecast = scale(X, XForecast)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    
    if classification:
        model = sklearn.svm.SVC(C=float(C), kernel=kernel, degree=int(degree), gamma=float(gamma))
    else:
        model = sklearn.svm.SVR(C=float(C), kernel=kernel, degree=int(degree), gamma=float(gamma))

    model.fit(x_train, y_train)

    dirname = os.path.dirname(__file__)
    
    f = open(os.path.join(dirname, "SVM model log.txt"), "w")
    f.write("Artificial R^2 with train/test split")
    f.write('\n')
    f.write(str(model.score(x_test, y_test)))
    f.close()

    model.fit(X, Y)
    predicted = model.predict(XForecast)
    return predicted
    

if __name__ == "__main__":
   xw.Book("myproject.xlsm").set_mock_caller()
   main()
